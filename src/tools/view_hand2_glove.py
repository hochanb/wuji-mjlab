"""Grasp sets on the Wuji Hand 2 with the glove covers: which grasps the covers
would push into the object.

Every grasp of a set (object pose in the palm frame + finger joints) is put on
the training hand (capsule collision model) with and without the glove covers
(``hand2_model.apply_hand2_glove``); MuJoCo's contacts give the penetration
depth per hand body. Each grasp is then held for 1 s like the training
warmup (finger targets at the grasp joints plus the warmup squeeze, palm as
in training, gravity on) on the bare hand, the glove hand with soft covers
(HAND2_GLOVE_SOLIMP) and with covers as hard as the links: how far the
object is pushed off its grasp pose. The viser viewer shows one grasp at a
time (initial or held state), the penetrating contacts marked, plus the
set's summary. Hold results are cached in tmp/glove_hold/.

Usage (repo root):
  pixi run python src/tools/view_hand2_glove.py \\
    --set cube=src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube/grasps/cube_dex_3k/all \\
    [--set stapler=<grasp folder>] [--port 8082]

``--set <compound object>=<grasp folder or grasps.npz>``; the object is a
repo compound object (``cube``, ``stapler``). The viewer binds to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from wuji_mjlab.assets.objects.inhand_object.compound_objects import compound_spec
from wuji_mjlab.assets.robots.wuji_hand2.hand2_model import hand2_glove_covers
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import (
  _get_spec,
  wuji_hand2_xml_path,
)
from wuji_mjlab.grasp_synth.synth import TRAINING_WRIST_QUAT_W
from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
  _HAND_CONDIM,
  _HAND_SLIDING_FRICTION,
  _WARMUP_SQUEEZE,
  _hand_contact_spec,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  get_hand2_reorient_spec,
)

PALM = "right_palm_link"
NUM_JOINTS = 20
HAND_RGBA = (0.78, 0.78, 0.8)
COVER_RGBA = (0.08, 0.08, 0.08)
OBJECT_RGBA = (0.9, 0.62, 0.15)
DEPTH_LEVELS_MM = (1.0, 3.0, 5.0)
VARIANTS = ("bare", "glove_soft", "glove_hard")
HOLD_S = (0.4, 1.0)  # warmup end, 1 s
CACHE = Path(__file__).resolve().parents[2] / "tmp" / "glove_hold_v3"  # gitignored


def build_model(object_name: str, variant: str) -> mujoco.MjModel:
  """The training hand with the grasp-stage contacts (condim 6, sliding
  friction 1.5: the middle of the training range) and the reorient task's
  physics (palm at the origin,
  gravity as in training: palm up, pitched by the mount) and the object on a
  free joint (a fixed object would be welded to the world like the palm, and
  MuJoCo never collides geoms of the same weld). ``variant``: bare /
  glove_soft / glove_hard (covers with the links' contact softness)."""
  spec = _hand_contact_spec(
    partial(
      get_hand2_reorient_spec,
      partial(_get_spec, wuji_hand2_xml_path("right")),
      glove=variant != "bare",
    ),
    _HAND_CONDIM,
    sum(_HAND_SLIDING_FRICTION) / 2,
  )
  palm = spec.body(PALM)
  palm.pos, palm.quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
  down = np.empty(3)
  mujoco.mju_rotVecQuat(down, [0.0, 0.0, -9.81], _conj(TRAINING_WRIST_QUAT_W))
  spec.option.gravity = down
  src = compound_spec(object_name).body("cube")
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_free")
  for g in src.geoms:
    body.add_geom(
      name=f"object_{g.name}",
      type=g.type,
      size=g.size,
      pos=g.pos,
      quat=g.quat,
      contype=g.contype,
      conaffinity=g.conaffinity,
      condim=g.condim,
      friction=g.friction,
      solref=g.solref,
      solimp=g.solimp,
      priority=g.priority,
      group=g.group,
      rgba=g.rgba,
      density=0,
      mass=g.mass if (g.contype or g.conaffinity) else 0.0,
    )
  model = spec.compile()
  assert model.nq == NUM_JOINTS + 7
  # The training env's solver options (mjlab MujocoCfg overrides the spec's)
  # with elliptic friction cones and impratio 10: with the default pyramidal
  # cones the grasps slip far more, and worse with more friction.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.iterations, model.opt.ls_iterations = 10, 20
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model.opt.impratio = 10.0
  if variant == "glove_hard":
    link = model.geom("right_palm_collision").id
    for name in hand2_glove_covers()["covers"]:
      model.geom_solimp[model.geom(f"{name}_col").id] = model.geom_solimp[link]
  return model


def _conj(q) -> np.ndarray:
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _squeeze(model) -> np.ndarray:
  """Warmup squeeze per finger joint (grasp_stage._WARMUP_SQUEEZE)."""
  out = np.zeros(NUM_JOINTS)
  for j in range(NUM_JOINTS):
    name = model.joint(j).name
    out[j] = next((v for p, v in _WARMUP_SQUEEZE.items() if re.fullmatch(p, name)), 0.0)
  return out


@dataclass
class Grasps:
  object_pos: np.ndarray
  object_quat: np.ndarray
  joint_q: np.ndarray
  ids: np.ndarray


def load_grasps(path: Path) -> Grasps:
  path = path / "grasps.npz" if path.is_dir() else path
  z = np.load(path)
  if str(z["frame_object_pose"]) != "wrist":
    raise ValueError(
      f"{path}: object pose frame {z['frame_object_pose']}, expected wrist"
    )
  return Grasps(z["object_pos"], z["object_quat"], z["joint_q"], z["ids"])


def set_grasp(model, data, g: Grasps, i: int) -> None:
  data.qpos[:NUM_JOINTS] = g.joint_q[i]
  data.qpos[NUM_JOINTS : NUM_JOINTS + 3] = g.object_pos[i]
  data.qpos[NUM_JOINTS + 3 :] = g.object_quat[i]
  mujoco.mj_forward(model, data)


def object_contacts(model, data) -> list[tuple[str, float, np.ndarray]]:
  """(hand body, penetration [m] > 0, position) of each hand-object contact."""
  obj = model.body("object").id
  out = []
  for c in data.contact[: data.ncon]:
    b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
    if obj not in (b1, b2) or c.dist >= 0:
      continue
    hand = b2 if b1 == obj else b1
    out.append((model.body(hand).name, -float(c.dist), c.pos.copy()))
  return out


@dataclass
class SetStats:
  cover_depth: np.ndarray  # [G] deepest cover penetration [m] (glove model)
  link_depth_glove: np.ndarray  # [G] deepest bare-link penetration, glove model
  link_depth_bare: np.ndarray  # [G] deepest penetration of the bare hand
  per_cover: dict[str, np.ndarray]  # cover -> [G] depth


def evaluate(bare, glove, g: Grasps) -> SetStats:
  covers = list(hand2_glove_covers()["covers"])
  n = len(g.ids)
  st = SetStats(np.zeros(n), np.zeros(n), np.zeros(n), {c: np.zeros(n) for c in covers})
  db, dg = mujoco.MjData(bare), mujoco.MjData(glove)
  for i in range(n):
    set_grasp(bare, db, g, i)
    for _, depth, _ in object_contacts(bare, db):
      st.link_depth_bare[i] = max(st.link_depth_bare[i], depth)
    set_grasp(glove, dg, g, i)
    for body, depth, _ in object_contacts(glove, dg):
      if body in st.per_cover:
        st.per_cover[body][i] = max(st.per_cover[body][i], depth)
        st.cover_depth[i] = max(st.cover_depth[i], depth)
      else:
        st.link_depth_glove[i] = max(st.link_depth_glove[i], depth)
  return st


_MODELS: dict = {}


def _hold_chunk(job) -> np.ndarray:
  """[n, len(HOLD_S), nq] qpos of grasps held with targets = joints + squeeze."""
  object_name, variant, q, pos, quat = job
  key = (object_name, variant)
  if key not in _MODELS:
    _MODELS[key] = build_model(object_name, variant)
  m = _MODELS[key]
  d = mujoco.MjData(m)
  lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
  squeeze = _squeeze(m)
  marks = [round(t / m.opt.timestep) for t in HOLD_S]
  out = np.zeros((len(q), len(HOLD_S), m.nq))
  for i in range(len(q)):
    mujoco.mj_resetData(m, d)
    d.qpos[:NUM_JOINTS] = q[i]
    d.qpos[NUM_JOINTS : NUM_JOINTS + 3] = pos[i]
    d.qpos[NUM_JOINTS + 3 :] = quat[i]
    d.ctrl[:] = np.clip(q[i] + squeeze, lo, hi)
    for step in range(1, marks[-1] + 1):
      mujoco.mj_step(m, d)
      if step in marks:
        out[i, marks.index(step)] = d.qpos
  return out


def hold(
  object_name: str, g: Grasps, tag: str, workers: int = 32
) -> dict[str, np.ndarray]:
  """{variant: [G, len(HOLD_S), nq]} held states (cached per set)."""
  path = CACHE / f"{tag}.npz"
  if path.exists():
    z = np.load(path)
    if all(v in z.files for v in VARIANTS) and z[VARIANTS[0]].shape[0] == len(g.ids):
      return {v: z[v] for v in VARIANTS}
  chunks = np.array_split(np.arange(len(g.ids)), workers * 4)
  out = {}
  with mp.get_context("fork").Pool(workers) as pool:
    for v in VARIANTS:
      jobs = [
        (object_name, v, g.joint_q[c], g.object_pos[c], g.object_quat[c])
        for c in chunks
      ]
      out[v] = np.concatenate(pool.map(_hold_chunk, jobs))
  CACHE.mkdir(parents=True, exist_ok=True)
  np.savez(path, **out)
  return out


def displacement(g: Grasps, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Object position [mm] and orientation [deg] change from the grasp pose."""
  dpos = (
    np.linalg.norm(qpos[:, NUM_JOINTS : NUM_JOINTS + 3] - g.object_pos, axis=1) * 1000
  )
  q1, q0 = qpos[:, NUM_JOINTS + 3 :], g.object_quat
  dot = np.abs(
    np.sum(q1 * q0, axis=1) / np.linalg.norm(q1, axis=1) / np.linalg.norm(q0, axis=1)
  )
  return dpos, np.degrees(2 * np.arccos(np.clip(dot, -1, 1)))


def _one(g: Grasps, i: int) -> Grasps:
  return Grasps(
    g.object_pos[i : i + 1],
    g.object_quat[i : i + 1],
    g.joint_q[i : i + 1],
    g.ids[i : i + 1],
  )


def hold_summary(g: Grasps, st: SetStats, held: dict[str, np.ndarray]) -> str:
  lines = [
    "",
    "**Held like the training warmup** (object moved from the grasp pose)",
    "",
  ]
  lines.append("| hand | at | pos med | pos > 5 mm | > 20 mm (lost) | rot med |")
  lines.append("|---|---|---|---|---|---|")
  for v in VARIANTS:
    for k, t in enumerate(HOLD_S):
      dp, dr = displacement(g, held[v][:, k])
      lines.append(
        f"| {v} | {t:g} s | {np.median(dp):.1f} mm | {np.mean(dp > 5):.1%} | {np.mean(dp > 20):.1%} | {np.median(dr):.1f} deg |"
      )
  lines += [
    "",
    "1 s, by initial cover depth: object moved (median) and how far the glove",
    "puts it from where the bare hand does (median, share > 5 mm):",
    "",
  ]
  lines.append(
    "| cover depth | grasps | moved bare / soft / hard | soft vs bare | hard vs bare |"
  )
  lines.append("|---|---|---|---|---|")
  depth = st.cover_depth * 1000
  moved = {v: displacement(g, held[v][:, -1])[0] for v in VARIANTS}
  bare_pos = held["bare"][:, -1, NUM_JOINTS : NUM_JOINTS + 3]
  apart = {
    v: np.linalg.norm(held[v][:, -1, NUM_JOINTS : NUM_JOINTS + 3] - bare_pos, axis=1)
    * 1000
    for v in VARIANTS[1:]
  }
  bins = ((None, None), (0, 1), (1, 3), (3, 5), (5, 10), (10, 99), (0, 99))
  for lo, hi in bins:
    if lo is None:
      sel, label = depth <= 0, "none"
    elif (lo, hi) == (0, 99):
      sel, label = np.ones_like(depth, bool), "all"
    else:
      sel, label = (depth > lo) & (depth <= hi), f"{lo:g}-{hi:g} mm"
    if not sel.any():
      continue
    mv = " / ".join(f"{np.median(moved[v][sel]):.1f}" for v in VARIANTS)
    ap = [
      f"{np.median(apart[v][sel]):.1f} mm, {np.mean(apart[v][sel] > 5):.0%}"
      for v in VARIANTS[1:]
    ]
    lines.append(f"| {label} | {sel.sum()} | {mv} mm | {ap[0]} | {ap[1]} |")
  return "\n".join(lines)


def summary(name: str, st: SetStats) -> str:
  n = len(st.cover_depth)
  lines = [f"**{name}**: {n} grasps"]
  for mm in DEPTH_LEVELS_MM:
    lines.append(
      f"- covers > {mm:g} mm into the object: {np.mean(st.cover_depth > mm / 1000):.1%}"
      f" (bare hand > {mm:g} mm: {np.mean(st.link_depth_bare > mm / 1000):.1%})"
    )
  hit = st.cover_depth > 0
  if hit.any():
    lines.append(
      f"- cover depth where touching: median {np.median(st.cover_depth[hit]) * 1000:.1f} mm, max {st.cover_depth.max() * 1000:.1f} mm"
    )
  ranked = sorted(st.per_cover.items(), key=lambda kv: -np.mean(kv[1] > 0.001))
  lines.append("- covers > 1 mm, share of grasps:")
  for cover, d in ranked[:8]:
    if np.mean(d > 0.001) > 0:
      lines.append(f"  - {cover.removeprefix('right_')}: {np.mean(d > 0.001):.1%}")
  return "\n".join(lines)


# --- viser ---------------------------------------------------------------------

STATES = (
  "grasp (initial)",
  "held 1 s: glove soft",
  "held 1 s: glove hard",
  "held 1 s: bare",
)
STATES_VARIANT = dict(
  zip(STATES, (None, "glove_soft", "glove_hard", "bare"), strict=True)
)
SIM_HANDS = {
  "glove, soft covers": "glove_soft",
  "glove, hard covers": "glove_hard",
  "bare": "bare",
}
ORDERS = (
  "cover depth (deepest first)",
  "glove soft: object moved (most first)",
  "grasp id",
)


def _geom_trimesh(model, g: int) -> trimesh.Trimesh | None:
  t, size = model.geom_type[g], model.geom_size[g]
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mid = model.geom_dataid[g]
    v = model.mesh_vert[
      model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
    ]
    f = model.mesh_face[
      model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]
    ]
    return trimesh.Trimesh(v, f, process=False)
  if t == mujoco.mjtGeom.mjGEOM_BOX:
    return trimesh.creation.box(extents=2 * size)
  if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
    return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
  if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[16, 16])
  if t == mujoco.mjtGeom.mjGEOM_SPHERE:
    return trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  return None


def _wxyz(mat: np.ndarray) -> np.ndarray:
  q = np.empty(4)
  mujoco.mju_mat2Quat(q, mat)
  return q


class Viewer:
  def __init__(
    self,
    server: viser.ViserServer,
    sets: dict[str, tuple[mujoco.MjModel, mujoco.MjModel, Grasps, SetStats, dict]],
  ):
    self.server, self.sets = server, sets
    # Client events run callbacks on a thread pool (one MjData: one update at a
    # time); setting a GUI value from Python runs its callbacks right away in
    # the same thread (rank reset in _select_set), hence reentrant.
    self.lock = threading.RLock()
    self.handles: list = []
    self.contact_handles: list = []
    gui = server.gui
    self.set_dd = gui.add_dropdown("Grasp set", tuple(sets))
    self.state_dd = gui.add_dropdown("State", STATES)
    self.order_dd = gui.add_dropdown("Order", ORDERS)
    self.rank = gui.add_slider("Rank", min=0, max=1, step=1, initial_value=0)
    self.show_covers = gui.add_checkbox("Show covers", True)
    self.show_collision = gui.add_checkbox("Collision geoms (training)", False)
    self.show_contacts = gui.add_checkbox("Mark penetrations", True)
    with gui.add_folder("Live physics"):
      self.sim_hand = gui.add_dropdown("Hand", tuple(SIM_HANDS))
      self.sim_run = gui.add_checkbox("Run (from the grasp)", False)
      self.sim_speed = gui.add_slider(
        "Speed (x real time)", min=0.05, max=2.0, step=0.05, initial_value=1.0
      )
      self.sim_restart = gui.add_button("Restart from the grasp")
      self.sim_info = gui.add_markdown("")
    self.info = gui.add_markdown("")
    self.stats = gui.add_markdown("")
    self.sim_models: dict = {}
    self.sim_data = None
    self.sim_run.on_update(lambda _: self._locked(self._sim_toggle))
    self.sim_hand.on_update(lambda _: self._locked(self._sim_toggle))
    self.sim_restart.on_click(lambda _: self._locked(self._sim_reset))
    threading.Thread(target=self._sim_loop, daemon=True).start()
    for h in (self.set_dd, self.order_dd):
      h.on_update(lambda _: self._locked(self._select_set))
    for h in (
      self.state_dd,
      self.rank,
      self.show_covers,
      self.show_collision,
      self.show_contacts,
    ):
      h.on_update(lambda _: self._locked(self._draw))
    self._select_set()

  def _order(self) -> np.ndarray:
    _, _, g, st, held = self.sets[self.set_dd.value]
    if self.order_dd.value == ORDERS[0]:
      return np.argsort(-st.cover_depth, kind="stable")
    if self.order_dd.value == ORDERS[1]:
      return np.argsort(-displacement(g, held["glove_soft"][:, -1])[0], kind="stable")
    return np.argsort(g.ids, kind="stable")

  def _locked(self, fn) -> None:
    with self.lock:
      fn()

  def _select_set(self) -> None:
    _, glove, g, st, held = self.sets[self.set_dd.value]
    self.sim_data = None  # restarted below on the new set's grasp
    self.rank.max = len(g.ids) - 1
    self.rank.value = 0
    self.stats.content = summary(self.set_dd.value, st) + hold_summary(g, st, held)
    self.display_model = glove
    self._build_scene(glove)
    if self.sim_run.value:
      self._sim_reset()
    else:
      self._draw()

  # --- live physics -------------------------------------------------------------

  def _grasp_index(self) -> int:
    return int(self._order()[int(self.rank.value)])

  def _sim_model(self) -> mujoco.MjModel:
    obj = self.set_dd.value.split(":")[0]
    key = (obj, SIM_HANDS[self.sim_hand.value])
    if key not in self.sim_models:
      self.sim_models[key] = build_model(*key)
    return self.sim_models[key]

  def _sim_toggle(self) -> None:
    if self.sim_run.value:
      self._sim_reset()
    else:
      self.sim_data = None
      self._build_scene(self.display_model)
      self._draw()

  def _sim_reset(self) -> None:
    """Start the live simulation at the current grasp: finger targets at the
    grasp joints plus the warmup squeeze (like ``hold``)."""
    if not self.sim_run.value:
      return
    m = self._sim_model()
    if self.model is not m:
      self._build_scene(m)
    g = self.sets[self.set_dd.value][2]
    i = self._grasp_index()
    d = mujoco.MjData(m)
    set_grasp(m, d, g, i)
    d.ctrl[:] = np.clip(
      g.joint_q[i] + _squeeze(m), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    )
    self.sim_data, self.data, self.sim_i = d, d, i
    self.sim_clock = time.time()
    self._render(SIM_HANDS[self.sim_hand.value])

  def _sim_loop(self) -> None:
    while True:
      time.sleep(1 / 30)
      with self.lock:
        d = self.sim_data
        if d is None:
          continue
        m = self.model
        now = time.time()
        steps = int((now - self.sim_clock) * self.sim_speed.value / m.opt.timestep)
        if steps <= 0:
          continue
        self.sim_clock += steps * m.opt.timestep / self.sim_speed.value
        for _ in range(min(steps, 20)):
          mujoco.mj_step(m, d)
        self._render(SIM_HANDS[self.sim_hand.value])

  def _build_scene(self, model) -> None:
    for h in self.handles:
      h.remove()
    self.handles = []
    self.model, self.data = model, mujoco.MjData(model)
    for g in range(model.ngeom):
      body = model.body(model.geom_bodyid[g]).name
      if body in ("r_mount", "right_wrist_tag_frame"):
        continue
      collides = bool(model.geom_contype[g] or model.geom_conaffinity[g])
      is_obj = body == "object"
      if is_obj and collides:
        kind = "object"
      elif is_obj:
        continue
      elif collides:
        kind = "collision"
      elif model.geom_group[g] == 1:
        kind = "cover" if "cover" in body else "hand"
      else:
        continue
      mesh = _geom_trimesh(model, g)
      if mesh is None:
        continue
      color, opacity = {
        "object": (OBJECT_RGBA, 0.55),
        "collision": ((0.3, 0.6, 1.0), 0.45),
        "cover": (COVER_RGBA, 0.75),
        "hand": (HAND_RGBA, 1.0),
      }[kind]
      h = self.server.scene.add_mesh_simple(
        f"/scene/{kind}/{g}",
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.uint32),
        color=color,
        opacity=opacity,
      )
      h.geom, h.kind, h.cover = g, kind, "cover" in body
      self.handles.append(h)

  def _draw(self) -> None:
    if self.sim_data is not None:
      if self._grasp_index() != self.sim_i:
        self._sim_reset()
      else:
        self._render(SIM_HANDS[self.sim_hand.value])
      return
    _, _, g, st, held = self.sets[self.set_dd.value]
    i = self._grasp_index()
    variant = STATES_VARIANT[self.state_dd.value]
    if variant is None:
      set_grasp(self.model, self.data, g, i)
    else:
      self.data.qpos[:] = held[variant][i, -1]
      mujoco.mj_forward(self.model, self.data)
    self._render(variant)

  def _render(self, variant: str | None) -> None:
    _, _, g, st, held = self.sets[self.set_dd.value]
    i = self._grasp_index()
    covers_on = self.show_covers.value and variant != "bare"
    for h in self.handles:
      h.position = self.data.geom_xpos[h.geom]
      h.wxyz = _wxyz(self.data.geom_xmat[h.geom])
      if h.kind == "collision":
        h.visible = self.show_collision.value and (covers_on or not h.cover)
      elif h.kind == "cover":
        h.visible = covers_on and not self.show_collision.value
      elif h.kind == "hand":
        h.visible = not self.show_collision.value
    for h in self.contact_handles:
      h.remove()
    self.contact_handles = []
    contacts = [
      c
      for c in object_contacts(self.model, self.data)
      if covers_on or "cover" not in c[0]
    ]
    if self.show_contacts.value:
      for k, (body, depth, pos) in enumerate(contacts):
        color = (1.0, 0.1, 0.1) if "cover" in body else (1.0, 0.8, 0.0)
        self.contact_handles.append(
          self.server.scene.add_icosphere(
            f"/contacts/{k}", radius=0.0015 + depth / 2, color=color, position=pos
          )
        )
    worst: dict[str, float] = {}
    for body, depth, _ in contacts:
      worst[body] = max(worst.get(body, 0.0), depth)
    rows = [
      f"| {b.removeprefix('right_')} | {d * 1000:.1f} |"
      for b, d in sorted(worst.items(), key=lambda kv: -kv[1])
    ]
    self.info.content = "\n".join(
      [
        f"**grasp id {int(g.ids[i])}** (rank {int(self.rank.value)})",
        f"cover depth {st.cover_depth[i] * 1000:.1f} mm; bare hand {st.link_depth_bare[i] * 1000:.1f} mm",
        "",
        "| held 1 s | object moved | turned |",
        "|---|---|---|",
        *[
          f"| {v} | {displacement(_one(g, i), held[v][i : i + 1, -1])[0][0]:.1f} mm | {displacement(_one(g, i), held[v][i : i + 1, -1])[1][0]:.1f} deg |"
          for v in VARIANTS
        ],
        "",
        "| body in the object | depth [mm] |",
        "|---|---|",
        *rows,
        "",
        "red: cover, yellow: bare link",
      ]
    )
    if self.sim_data is not None:
      d = self.sim_data
      dp, dr = displacement(_one(g, self.sim_i), d.qpos[None])
      self.sim_info.content = (
        f"t = {d.time:.2f} s; object moved {dp[0]:.1f} mm, turned {dr[0]:.1f} deg"
        f" from the grasp pose"
      )
    else:
      self.sim_info.content = ""


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument(
    "--set",
    action="append",
    required=True,
    help="<compound object>=<grasp folder or grasps.npz>",
  )
  ap.add_argument("--port", type=int, default=8082)
  args = ap.parse_args()
  sets = {}
  for item in args.set:
    obj, path = item.split("=", 1)
    g = load_grasps(Path(path))
    bare, glove = build_model(obj, "bare"), build_model(obj, "glove_soft")
    t = time.time()
    st = evaluate(bare, glove, g)
    folder = Path(path).parent if Path(path).name == "grasps.npz" else Path(path)
    tag = folder.parent.name if folder.name == "all" else folder.name
    name = f"{obj}: {tag}"
    held = hold(obj, g, f"{obj}_{tag}")
    sets[name] = (bare, glove, g, st, held)
    print(
      summary(name, st),
      hold_summary(g, st, held),
      f"\n({time.time() - t:.1f} s)",
      flush=True,
    )
  server = viser.ViserServer(host="127.0.0.1", port=args.port)
  up = -np.asarray(glove.opt.gravity) / np.linalg.norm(glove.opt.gravity)
  server.scene.set_up_direction(tuple(float(x) for x in up))
  Viewer(server, sets)
  print(f"viewer on http://127.0.0.1:{args.port}", flush=True)
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
