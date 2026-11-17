"""Grasp sets on the Wuji Hand 2 with the glove covers: which grasps the covers
would push into the object.

Every grasp of a set (object pose in the palm frame + finger joints) is put on
the training hand (capsule collision model) with and without the glove covers
(``hand2_model.apply_hand2_glove``); MuJoCo's contacts give the penetration
depth per hand body. The viser viewer shows one grasp at a time, the grasps
ordered by how deep the covers go into the object, with the penetrating
contacts marked, plus the set's summary.

Usage (repo root):
  pixi run python src/tools/view_hand2_glove.py \\
    --set cube=src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube/grasps/cube_dex_3k/all \\
    [--set stapler=<grasp folder>] [--port 8082]

``--set <compound object>=<grasp folder or grasps.npz>``; the object is a
repo compound object (``cube``, ``stapler``). The viewer binds to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from wuji_mjlab.assets.objects.inhand_object.compound_objects import compound_spec
from wuji_mjlab.assets.robots.wuji_hand2.hand2_model import (
  get_hand2_model_spec,
  hand2_glove_covers,
)
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import (
  _get_spec,
  wuji_hand2_xml_path,
)

PALM = "right_palm_link"
NUM_JOINTS = 20
HAND_RGBA = (0.78, 0.78, 0.8)
COVER_RGBA = (0.08, 0.08, 0.08)
OBJECT_RGBA = (0.9, 0.62, 0.15)
DEPTH_LEVELS_MM = (1.0, 3.0, 5.0)


def build_model(object_name: str, glove: bool) -> mujoco.MjModel:
  """The training hand (palm at the origin) and the object on a free joint
  (a fixed object would be welded to the world like the palm, and MuJoCo
  never collides geoms of the same weld)."""
  spec = get_hand2_model_spec(
    partial(_get_spec, wuji_hand2_xml_path("right")), glove=glove
  )
  palm = spec.body(PALM)
  palm.pos, palm.quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
  obj_spec = compound_spec(object_name)
  src = obj_spec.body("cube")
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
      group=g.group,
      rgba=g.rgba,
      density=0,
      mass=0.01,
    )
  model = spec.compile()
  assert model.nq == NUM_JOINTS + 7
  return model


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
    sets: dict[str, tuple[mujoco.MjModel, mujoco.MjModel, Grasps, SetStats]],
  ):
    self.server, self.sets = server, sets
    self.handles: list = []
    self.contact_handles: list = []
    gui = server.gui
    self.set_dd = gui.add_dropdown("Grasp set", tuple(sets))
    self.order_dd = gui.add_dropdown(
      "Order", ("cover depth (deepest first)", "grasp id")
    )
    self.rank = gui.add_slider("Rank", min=0, max=1, step=1, initial_value=0)
    self.show_covers = gui.add_checkbox("Show covers", True)
    self.show_collision = gui.add_checkbox("Collision geoms (training)", False)
    self.show_contacts = gui.add_checkbox("Mark penetrations", True)
    self.info = gui.add_markdown("")
    self.stats = gui.add_markdown("")
    for h in (self.set_dd, self.order_dd):
      h.on_update(lambda _: self._select_set())
    for h in (self.rank, self.show_covers, self.show_collision, self.show_contacts):
      h.on_update(lambda _: self._draw())
    self._select_set()

  def _order(self) -> np.ndarray:
    _, _, g, st = self.sets[self.set_dd.value]
    if self.order_dd.value.startswith("cover"):
      return np.argsort(-st.cover_depth, kind="stable")
    return np.argsort(g.ids, kind="stable")

  def _select_set(self) -> None:
    _, glove, g, st = self.sets[self.set_dd.value]
    self.rank.max = len(g.ids) - 1
    self.rank.value = 0
    self.stats.content = summary(self.set_dd.value, st)
    self._build_scene(glove)
    self._draw()

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
    _, _, g, st = self.sets[self.set_dd.value]
    i = int(self._order()[int(self.rank.value)])
    set_grasp(self.model, self.data, g, i)
    for h in self.handles:
      h.position = self.data.geom_xpos[h.geom]
      h.wxyz = _wxyz(self.data.geom_xmat[h.geom])
      if h.kind == "collision":
        h.visible = self.show_collision.value and (
          self.show_covers.value or not h.cover
        )
      elif h.kind == "cover":
        h.visible = self.show_covers.value and not self.show_collision.value
      elif h.kind == "hand":
        h.visible = not self.show_collision.value
    for h in self.contact_handles:
      h.remove()
    self.contact_handles = []
    contacts = object_contacts(self.model, self.data)
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
        "| body in the object | depth [mm] |",
        "|---|---|",
        *rows,
        "",
        "red: cover, yellow: bare link",
      ]
    )


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
    bare, glove = build_model(obj, False), build_model(obj, True)
    t = time.time()
    st = evaluate(bare, glove, g)
    name = f"{obj}: {Path(path).parent.name if Path(path).name == 'all' else Path(path).name}"
    sets[name] = (bare, glove, g, st)
    print(summary(name, st), f"\n({time.time() - t:.1f} s)", flush=True)
  server = viser.ViserServer(host="127.0.0.1", port=args.port)
  server.scene.set_up_direction("+z")
  Viewer(server, sets)
  print(f"viewer on http://127.0.0.1:{args.port}", flush=True)
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
