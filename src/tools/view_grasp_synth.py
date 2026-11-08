"""Generate grasps with ``wuji_mjlab.grasp_synth`` and show them all in viser.

Each grasp type is a grid: one row per object pose, one column per grasp,
the hand palm-up (``hand_quat_w``). A dropdown switches the grasp type. Every
grasp is simulated live like the synthesizer's hold test: the actuators hold
the squeeze targets, the object is pinned while the grip settles and then
released; the cycle restarts after ``hold`` seconds. GUI: play / pause,
speed, gravity on / off (palm-up), restart.
Results are cached in ``tmp/grasp_synth/<name>.npz`` (``--load`` to view
them again without generating).

Usage (repo root):
  pixi run python src/tools/view_grasp_synth.py [--mesh obj.obj --mass 0.1] \\
    [--types pinch,tripod,five_tip,power] [--k 4] [--poses 6] [--port 8090] [--load]

Without ``--mesh`` the object is the 54 mm reorient cube; poses are the cube
spawn point in the palm frame with the identity and random orientations.
The viewer binds to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjlab.sim import MujocoCfg
from wuji_mjlab.grasp_synth import (
  GRASP_TYPES,
  GraspObject,
  GraspSynthCfg,
  GraspSynthesizer,
)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "tmp" / "grasp_synth"  # gitignored
CUBE_POS = (0.0042, 0.0791, -0.0673)  # reorient cube spawn, palm frame
SPACING = 0.22
HAND_BODIES = ("right_palm_link", "right_finger")


def _poses(n: int, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  q = rng.normal(size=(n, 4))
  q[0] = (1.0, 0.0, 0.0, 0.0)
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  return np.concatenate([np.tile(CUBE_POS, (n, 1)), q], 1)


def _quat_mul(a, b):
  out = np.empty(4)
  mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
  return out


def _rotate(q, v):
  out = np.empty(3)
  mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
  return out


def generate(args, obj: GraspObject, cfg: GraspSynthCfg, poses) -> dict:
  syn = GraspSynthesizer(cfg)
  out = {"poses": poses}
  for t in args.types:
    t0 = time.time()
    r = syn.generate(obj, poses, t, args.k)
    print(
      f"{t}: {[len(q) for q in r.joint_pos]} grasps per pose, attempts "
      f"{r.attempts.tolist()}, {time.time() - t0:.1f} s"
    )
    for p in range(len(poses)):
      out[f"{t}/q{p}"] = r.joint_pos[p]
      out[f"{t}/target{p}"] = r.joint_target[p]
      out[f"{t}/res{p}"] = r.wrench_residual[p]
  return out


def _quat_mul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  )


class _Cell:
  """One grasp: its own MjData stepping the hold test."""

  def __init__(self, m, offset, q, target, pose, label):
    self.m, self.offset, self.q, self.target, self.pose = m, offset, q, target, pose
    self.label_text, self.label = label, None
    self.data = mujoco.MjData(m)
    self.obj_adr = int(m.joint("object_free").qposadr[0])
    self.obj_dof = int(m.joint("object_free").dofadr[0])

  def reset(self) -> None:
    d = self.data
    mujoco.mj_resetData(self.m, d)
    d.qpos[:20] = self.q
    d.ctrl[:] = self.target
    self.pin()
    mujoco.mj_forward(self.m, d)

  def pin(self) -> None:
    self.data.qpos[self.obj_adr : self.obj_adr + 7] = self.pose
    self.data.qvel[self.obj_dof : self.obj_dof + 6] = 0.0

  def drift(self) -> tuple[float, float]:
    q = self.data.qpos[self.obj_adr : self.obj_adr + 7]
    dot = min(1.0, abs(float(np.dot(q[3:], self.pose[3:]))))
    return float(np.linalg.norm(q[:3] - self.pose[:3])), float(
      np.degrees(2 * np.arccos(dot))
    )


def _cpu_model(obj: GraspObject, cfg: GraspSynthCfg) -> mujoco.MjModel:
  """The synthesizer's hand + object model with its physics options."""
  m = GraspSynthesizer(cfg)._build_model(obj)
  MujocoCfg(
    timestep=cfg.timestep,
    cone=cfg.cone,
    impratio=cfg.impratio,
    iterations=cfg.solver_iterations,
    ls_iterations=cfg.ls_iterations,
  ).apply(m)
  return m


def view(args, obj, cfg: GraspSynthCfg, data: dict) -> None:
  """``obj``: one GraspObject, or {group: GraspObject}. Optional data keys:
  ``<group>/pose<row>`` per-grasp object poses, ``row_names``."""
  objs = obj if isinstance(obj, dict) else dict.fromkeys(args.types, obj)
  models = {id(o): _cpu_model(o, cfg) for o in objs.values()}
  # Hand geoms come first and the object body last in every model.
  m = models[id(objs[args.types[0]])]
  poses = data["poses"]
  hand_q = np.asarray(cfg.hand_quat_w, float)
  hand_q /= np.linalg.norm(hand_q)
  hand_rot = np.empty(9)
  mujoco.mju_quat2Mat(hand_rot, hand_q)
  hand_rot = hand_rot.reshape(3, 3)
  gravity_palm = hand_rot.T @ np.array([0.0, 0.0, -9.81])
  obj_body = m.body("object").id
  geoms = np.array(
    [
      g
      for g in range(m.ngeom)
      if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      and m.geom_group[g] == 1
      and m.body(int(m.geom_bodyid[g])).name.startswith(HAND_BODIES)
    ]
  )

  server = viser.ViserServer(host="127.0.0.1", port=args.port)
  server.scene.set_up_direction("+z")
  groups: dict[str, dict] = {}
  for t in args.types:
    o, mt = objs[t], models[id(objs[t])]
    cells, empty = [], []
    for p in range(len(poses)):
      q_all, res = data[f"{t}/q{p}"], data[f"{t}/res{p}"]
      tgt = data.get(f"{t}/target{p}", q_all)  # old caches: no squeeze
      for g in range(len(q_all)):
        offset = np.array([g * SPACING, -p * SPACING, 0.0])
        row = data["row_names"][p] if "row_names" in data else f"pose {p}"
        label = f"{row} / grasp {g}  r={res[g]:.3f}"
        pose = data[f"{t}/pose{p}"][g] if f"{t}/pose{p}" in data else poses[p]
        cells.append(_Cell(mt, offset, q_all[g], tgt[g], pose, label))
      if len(q_all) == 0:
        row = data["row_names"][p] if "row_names" in data else f"pose {p}"
        empty.append((np.array([0.0, -p * SPACING, 0.0]), poses[p], row))
    handles: list = []
    for c in cells:
      c.reset()
      c.label = server.scene.add_label(
        f"/{t}/label_{id(c)}", c.label_text, position=c.offset + (0, 0, 0.13)
      )
      handles.append(c.label)
    for i, (offset, pose, row) in enumerate(empty):
      handles.append(
        server.scene.add_label(
          f"/{t}/empty_label{i}", f"{row}: no grasp", position=offset + (0, 0, 0.13)
        )
      )
      handles.append(
        server.scene.add_mesh_simple(
          f"/{t}/empty_object{i}",
          np.asarray(o.mesh.vertices),
          np.asarray(o.mesh.faces),
          color=(150, 150, 150),
          opacity=0.5,
          wxyz=_quat_mul(hand_q, pose[3:]),
          position=offset + _rotate(hand_q, pose[:3]),
        )
      )
    meshes = {}
    if cells:
      ones_q = np.tile([1.0, 0.0, 0.0, 0.0], (len(cells), 1))
      zeros_p = np.zeros((len(cells), 3))
      for g in geoms:
        mesh = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
        meshes[g] = server.scene.add_batched_meshes_simple(
          f"/{t}/geom{g}",
          m.mesh_vert[va : va + vn],
          m.mesh_face[fa : fa + fn],
          batched_wxyzs=ones_q,
          batched_positions=zeros_p,
          batched_colors=tuple(int(255 * c) for c in m.geom_rgba[g, :3]),
        )
      meshes["object"] = server.scene.add_batched_meshes_simple(
        f"/{t}/object",
        np.asarray(o.mesh.vertices),
        np.asarray(o.mesh.faces),
        batched_wxyzs=ones_q,
        batched_positions=zeros_p,
        batched_colors=(230, 150, 40),
        opacity=0.85,
      )
      # The object's local frame (mesh frame): x red, y green, z blue.
      meshes["object_axes"] = server.scene.add_batched_axes(
        f"/{t}/object_axes",
        batched_wxyzs=ones_q,
        batched_positions=zeros_p,
        axes_length=1.4 * o.radius,
        axes_radius=0.0015,
      )
    handles += [h for k, h in meshes.items() if k != "object_axes"]
    groups[t] = dict(cells=cells, meshes=meshes, handles=handles)

  def draw(t: str) -> None:
    cells, meshes = groups[t]["cells"], groups[t]["meshes"]
    if not cells:
      return
    offsets = np.array([c.offset for c in cells])
    gq = np.empty((len(cells), len(geoms), 4))
    gp = np.empty((len(cells), len(geoms), 3))
    oq = np.empty((len(cells), 4))
    op = np.empty((len(cells), 3))
    for i, c in enumerate(cells):
      d = c.data
      gp[i] = d.geom_xpos[geoms]
      for k, g in enumerate(geoms):
        mujoco.mju_mat2Quat(gq[i, k], d.geom_xmat[g])
      op[i] = d.xpos[obj_body]
      oq[i] = d.xquat[obj_body]
    gq = _quat_mul_batch(np.broadcast_to(hand_q, gq.shape), gq)
    gp = gp @ hand_rot.T + offsets[:, None]
    for k, g in enumerate(geoms):
      meshes[g].batched_wxyzs = gq[:, k]
      meshes[g].batched_positions = gp[:, k]
    meshes["object"].batched_wxyzs = _quat_mul_batch(
      np.broadcast_to(hand_q, oq.shape), oq
    )
    meshes["object"].batched_positions = op @ hand_rot.T + offsets
    meshes["object_axes"].batched_wxyzs = meshes["object"].batched_wxyzs
    meshes["object_axes"].batched_positions = meshes["object"].batched_positions

  gui = server.gui
  choice = gui.add_dropdown(
    "grasp type", tuple(args.types), initial_value=args.types[0]
  )
  playing = gui.add_checkbox("play", True)
  gravity = gui.add_checkbox("gravity (palm frame)", cfg.gravity > 0)
  speed = gui.add_slider("speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
  hold_s = gui.add_slider("hold [s]", min=0.5, max=10.0, step=0.5, initial_value=3.0)
  axes = gui.add_checkbox("object axes (x r, y g, z b)", True)
  restart = gui.add_button("restart")
  info = gui.add_markdown("")
  # GUI callbacks run on viser's thread: they only flag work for the loop,
  # which owns the MjData (stepping and resetting them concurrently crashes).
  state = {"step": 0, "show": True, "reset": False, "axes": False}

  def reset_all(_=None) -> None:
    state["step"] = 0
    for c in groups[choice.value]["cells"]:
      c.reset()

  def show(_=None) -> None:
    for t, grp in groups.items():
      for h in grp["handles"]:
        h.visible = t == choice.value
      if "object_axes" in grp["meshes"]:
        grp["meshes"]["object_axes"].visible = t == choice.value and axes.value
    reset_all()
    draw(choice.value)

  def set_gravity(_=None) -> None:
    for model in models.values():
      model.opt.gravity[:] = gravity_palm if gravity.value else 0.0

  choice.on_update(lambda _: state.update(show=True))
  restart.on_click(lambda _: state.update(reset=True))
  axes.on_update(lambda _: state.update(axes=True))
  gravity.on_update(set_gravity)
  set_gravity()
  print(f"viser: http://127.0.0.1:{args.port}", flush=True)

  dt, frame = m.opt.timestep, 1.0 / 30.0
  settle = cfg.settle_steps
  while True:
    start = time.time()
    if state["show"]:
      state["show"] = state["reset"] = False
      show()
    elif state["reset"]:
      state["reset"] = False
      reset_all()
    if state["axes"]:  # visibility only, the simulation keeps running
      state["axes"] = False
      for t, grp in groups.items():
        if "object_axes" in grp["meshes"]:
          grp["meshes"]["object_axes"].visible = t == choice.value and axes.value
    t = choice.value
    cells = groups[t]["cells"]
    if playing.value and cells:
      for _ in range(max(1, round(speed.value * frame / dt))):
        if state["step"] >= settle + round(hold_s.value / dt):
          reset_all()
        for c in cells:
          if state["step"] < settle:
            c.pin()
          mujoco.mj_step(c.m, c.data)
        state["step"] += 1
      draw(t)
      held = 0
      for c in cells:
        mm, deg = c.drift()
        ok = mm * 1e-3 <= cfg.max_pos_drift and np.radians(deg) <= cfg.max_rot_drift
        held += ok
        c.label.text = (
          f"{c.label_text}\n{'OK' if ok else 'MOVED'} {mm:.1f} mm {deg:.1f} deg"
        )
      phase = (
        "settling (object pinned)"
        if state["step"] < settle
        else f"holding {(state['step'] - settle) * dt:.1f} s"
      )
      info.content = (
        f"**{t}**: {len(cells)} grasps (rows: poses, columns: grasps)  \n"
        f"{phase}; within {cfg.max_pos_drift * 1e3:.0f} mm / "
        f"{np.degrees(cfg.max_rot_drift):.0f} deg: {held}/{len(cells)}"
      )
    time.sleep(max(0.0, frame - (time.time() - start)))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--mesh", type=Path, default=None)
  parser.add_argument("--mass", type=float, default=0.12)
  parser.add_argument("--types", default=",".join(GRASP_TYPES))
  parser.add_argument("--k", type=int, default=4)
  parser.add_argument("--poses", type=int, default=6)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--name", default=None, help="cache name (default: object)")
  parser.add_argument("--port", type=int, default=8090)
  parser.add_argument("--load", action="store_true", help="view the cached result")
  args = parser.parse_args()
  args.types = args.types.split(",")

  if args.mesh:
    obj = GraspObject.from_file(args.mesh, mass=args.mass)
  else:
    obj = GraspObject.box((0.027, 0.027, 0.027), mass=args.mass)
  cfg = GraspSynthCfg()
  path = CACHE / f"{args.name or (args.mesh.stem if args.mesh else 'cube')}.npz"
  if args.load:
    data = dict(np.load(path))
  else:
    data = generate(args, obj, cfg, _poses(args.poses, args.seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f"saved {path}")
  view(args, obj, cfg, data)


if __name__ == "__main__":
  main()
