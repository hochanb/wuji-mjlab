"""Grasp-stage sets of Dexonomy grasps (grasp-to-grasp training).

Two steps:

1. ``synth``: Dexonomy grasps of one object for every library grasp type (as
   many as each type yields, ``--k`` at most per type and seed), run with
   ``generate_many``; repeated per ``--seeds`` and merged into one pool
   (``.npz``: object pose in the wrist frame, joints, squeeze targets, type).
2. ``sets``: N grasps drawn at random from one or more pools, converted to the
   grasp-stage format and written as grasp sets plus training files:

   - ``<name>/all``: the N grasps (``--many-to-many``: init = goal = all,
     one stage, ``trainings/<name>_m2m.yaml``);
   - ``--many-to-one K``: one goal grasp (``<name>/goal``; the medoid of the
     object poses, or ``--goal-index``) and the other grasps as initial
     states in K cumulative stages ``<name>/init_stage1..K``, nearest object
     pose to the goal first (``trainings/<name>_m2o_curriculum.yaml``).

Grasp-stage fields: object pose in the palm frame and finger joints from the
grasp (``joint_pos``, just touching); tip targets are the fingertip pad
reference points (``HAND2_TIP_PAD_POINTS``) projected onto the object
surface, with the outward normal there. Weights: pose 1; joints 1 for the
fingers touching the object (``--joint-weights all``: every finger; the
others stay where the synthesis started them, extended); tips 1 for the
fingers whose pad (``tip_sensor`` link) touches the object (else 0;
``--tip-weights none``: all 0, so rewards and success use the hand shape).

Usage (repo root):
  OBJ=src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube
  pixi run python src/tools/make_dexonomy_grasp_sets.py synth --object cube \\
    --k 400 --seeds 0 --out tmp/dex_cube_pool.npz
  pixi run python src/tools/make_dexonomy_grasp_sets.py sets \\
    --pool tmp/dex_cube_pool.npz --object-dir $OBJ --name cube_dex_3k \\
    --count 3000 --many-to-one 5 --many-to-many --tip-weights none \\
    --joint-weights contact --training
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

HAND = "wuji_hand2_right"


def _quat_mat(q: np.ndarray) -> np.ndarray:
  w, x, y, z = np.moveaxis(np.asarray(q, np.float64), -1, 0)
  return np.stack(
    (
      np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
      np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
      np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
    ),
    -2,
  )


def _rot_dist(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
  """Angle [rad] between orientations (broadcast, wxyz)."""
  dot = np.abs((np.asarray(qa) * np.asarray(qb)).sum(-1))
  return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _object(name: str):
  from wuji_mjlab.grasp_synth import GraspObject

  return GraspObject.from_compound(name)


# -- synth ---------------------------------------------------------------------
def synth(args) -> None:
  from wuji_mjlab.grasp_synth import DexonomyCfg, DexonomySynthesizer, GraspSynthCfg

  obj = _object(args.object)
  print(f"{args.object}: mass {obj.mass:.3f} kg, extents {obj.mesh.extents}")
  pools = []
  for seed in args.seeds:
    dex = DexonomySynthesizer(
      DexonomyCfg(max_rounds=args.max_rounds, synth=GraspSynthCfg(seed=seed))
    )
    types = args.types or list(dex.library)
    t0 = time.time()
    res = dex.generate_many(obj, types, k=args.k)
    print(f"seed {seed}: {time.time() - t0:.0f} s")
    for t, r in res.items():
      n = len(r.joint_pos)
      print(f"  {t:28s} {n:4d} / {args.k} ({r.attempts} attempts)")
      if n:
        pools.append(
          {
            "object_pose": r.object_pose,
            "joint_pos": r.joint_pos,
            "joint_target": r.joint_target,
            "wrench_residual": r.wrench_residual,
            "grasp_type": np.full(n, t),
            "seed": np.full(n, seed),
          }
        )
  pool = {k: np.concatenate([p[k] for p in pools]) for k in pools[0]}
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, object=np.str_(args.object), **pool)
  types, counts = np.unique(pool["grasp_type"], return_counts=True)
  print(f"{out}: {len(pool['joint_pos'])} grasps, {len(types)} types")


# -- sets ----------------------------------------------------------------------
def _load_pools(paths) -> dict:
  pools = [dict(np.load(p)) for p in paths]
  names = {str(p["object"]) for p in pools}
  if len(names) != 1:
    raise SystemExit(f"pools of different objects: {sorted(names)}")
  keys = [k for k in pools[0] if k != "object"]
  out = {k: np.concatenate([p[k] for p in pools]) for k in keys}
  out["object"] = names.pop()
  return out


def _dedupe(pool: dict, pos_gap: float, rot_gap: float) -> np.ndarray:
  """Rows kept: per grasp type, object poses at least the gaps apart (pools
  of different seeds can repeat a grasp)."""
  keep = []
  pose = pool["object_pose"]
  for t in np.unique(pool["grasp_type"]):
    kept: list[int] = []
    for i in np.flatnonzero(pool["grasp_type"] == t):
      if kept:
        k = np.array(kept)
        near = (np.linalg.norm(pose[k, :3] - pose[i, :3], axis=-1) < pos_gap) & (
          _rot_dist(pose[k, 3:], pose[i, 3:]) < rot_gap
        )
        if near.any():
          continue
      kept.append(i)
    keep += kept
  return np.sort(np.array(keep))


def _nearest_surface(mesh, pts: np.ndarray, chunk: int = 512):
  """Closest surface point, its face normal and the signed distance (< 0
  inside, by that normal) per point; brute force over all faces (no rtree)."""
  from trimesh.triangles import closest_point

  tri, fn = mesh.triangles, mesh.face_normals
  out_p, out_n, out_d = [], [], []
  for s in range(0, len(pts), chunk):
    p = pts[s : s + chunk]
    c = closest_point(np.tile(tri, (len(p), 1, 1)), np.repeat(p, len(tri), 0))
    c = c.reshape(len(p), len(tri), 3)
    best = np.linalg.norm(c - p[:, None], axis=-1).argmin(1)
    cp, n = c[np.arange(len(p)), best], fn[best]
    out_p.append(cp)
    out_n.append(n)
    out_d.append(((p - cp) * n).sum(-1))
  return np.concatenate(out_p), np.concatenate(out_n), np.concatenate(out_d)


def _tip_targets(obj, object_pose: np.ndarray, joint_pos: np.ndarray):
  """Fingertip pad reference points (palm frame, by FK) projected onto the
  object surface (points, outward normals; object frame); per finger
  whether its pad touches the object and whether any of its links does
  (within 2 mm)."""
  import mujoco
  from wuji_mjlab.grasp_synth import GraspSynthCfg, GraspSynthesizer
  from wuji_mjlab.grasp_synth.templates import template_from_grasp
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
    HAND2_TIP_PAD_POINTS,
  )

  syn = GraspSynthesizer(GraspSynthCfg(num_worlds=64))
  syn._setup(obj, 64)
  model = syn.model
  data = mujoco.MjData(model)
  palm = model.body("right_palm_link").id
  sites = [model.site(f"right_finger{f}_tip").id for f in range(1, 6)]
  pads = np.asarray(HAND2_TIP_PAD_POINTS)
  g = len(joint_pos)
  ref = np.zeros((g, 5, 3))
  touch = np.zeros((g, 5), bool)
  pad = np.zeros((g, 5), bool)
  for i in range(g):
    tpl = template_from_grasp(model, "", joint_pos[i], object_pose[i])
    touch[i, [f - 1 for f in tpl.fingers]] = True
    bodies = {c.body for c in tpl.contacts}
    pad[i] = [f"right_finger{f}_tip_sensor" in bodies for f in range(1, 6)]
    data.qpos[:20] = joint_pos[i]
    mujoco.mj_kinematics(model, data)
    rp, rm = data.xpos[palm], data.xmat[palm].reshape(3, 3)
    for f, s in enumerate(sites):
      p = data.site_xpos[s] + data.site_xmat[s].reshape(3, 3) @ pads[f]
      ref[i, f] = rm.T @ (p - rp)  # palm frame
  rot = _quat_mat(object_pose[:, 3:])  # object -> palm
  ref_obj = np.einsum("gji,gfj->gfi", rot, ref - object_pose[:, None, :3])
  points, normals, _ = _nearest_surface(obj.mesh, ref_obj.reshape(-1, 3))
  return points.reshape(g, 5, 3), normals.reshape(g, 5, 3), pad, touch


def _grasp_set(pool: dict, rows: np.ndarray, tips, joint_w: str, tip_w: str):
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import GraspSet

  points, normals, pad, touch = tips
  g = len(rows)
  return GraspSet(
    ids=rows.astype(np.int64),  # row in the merged pool
    object_pos=pool["object_pose"][rows, :3].astype(np.float32),
    object_quat=pool["object_pose"][rows, 3:].astype(np.float32),
    pose_weights=np.ones((g, 2), np.float32),
    tip_points=points[rows].astype(np.float32),
    tip_normals=normals[rows].astype(np.float32),
    tip_weights=(pad[rows] & (tip_w == "pad")).astype(np.float32),
    joint_q=pool["joint_pos"][rows].astype(np.float32),
    joint_weights=(touch[rows] | (joint_w == "all")).astype(np.float32),
  )


def _pose_dist(pose: np.ndarray, ref: np.ndarray, pos_scale: float, rot_scale: float):
  return (
    np.linalg.norm(pose[..., :3] - ref[..., :3], axis=-1) / pos_scale
    + _rot_dist(pose[..., 3:], ref[..., 3:]) / rot_scale
  )


def _stage(name, obj, init, goal, iterations) -> str:
  return (
    f"  - name: {name}\n"
    f"    objects: [{obj}]\n"
    f"    init_grasps: {{{obj}: [{init}]}}\n"
    f"    goal_grasps: {{{obj}: [{goal}]}}\n"
    f"    advance: {{after_iterations: {iterations}}}\n"
  )


def _training(header: str, stages: list[str]) -> str:
  return (
    f"{header}"
    "version: 1\n"
    f"hand: {HAND}\n"
    "encoding: {name: none, dim: 0}\n"
    "object_root: ../objects\n"
    "stages:\n" + "".join(stages)
  )


def sets(args) -> None:
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
    hand_joint_limits,
    load_training,
    save_grasp_set_npz,
  )

  pool = _load_pools(args.pool)
  obj_name = pool["object"]
  n_all = len(pool["joint_pos"])
  keep = _dedupe(pool, args.dedupe_pos, math.radians(args.dedupe_rot))
  # Goal grasps need every joint (w_joint = 1) within the soft limits.
  _, _, soft = hand_joint_limits(HAND)
  q = pool["joint_pos"][keep]
  keep = keep[((q >= soft[:, 0] - 1e-6) & (q <= soft[:, 1] + 1e-6)).all(-1)]
  print(f"pool: {n_all} grasps, {len(keep)} after dedupe / soft-limit check")
  rng = np.random.default_rng(args.seed)
  if args.count > len(keep):
    print(f"only {len(keep)} grasps: using all of them")
  rows = np.sort(rng.choice(keep, min(args.count, len(keep)), replace=False))
  tips = [
    np.zeros((n_all, 5, 3)),
    np.zeros((n_all, 5, 3)),
    np.zeros((n_all, 5), bool),
    np.zeros((n_all, 5), bool),
  ]
  for a, b in zip(
    tips,
    _tip_targets(_object(obj_name), pool["object_pose"][rows], pool["joint_pos"][rows]),
    strict=True,
  ):
    a[rows] = b
  everything = _grasp_set(pool, rows, tips, args.joint_weights, args.tip_weights)
  types, counts = np.unique(pool["grasp_type"][rows], return_counts=True)
  print(f"{len(rows)} grasps of {len(types)} types:")
  for t, c in sorted(zip(types, counts, strict=True), key=lambda x: -x[1]):
    print(f"  {t:28s} {c:4d}")
  touching = everything.tip_weights.sum(-1)
  print(
    f"grasps by number of w_tip > 0: {np.bincount(touching.astype(int), minlength=6)}"
  )
  print(
    f"w_joint = 1 per finger (thumb..pinky): {everything.joint_weights.mean(0).round(2)}"
  )

  obj_dir = Path(args.object_dir)
  root = obj_dir / "grasps" / args.name
  rel = f"grasps/{args.name}"
  save_grasp_set_npz(root / "all", everything, HAND)
  np.savez(
    root / "meta.npz",
    pool_row=rows,
    grasp_type=pool["grasp_type"][rows],
    joint_target=pool["joint_target"][rows],
  )
  trainings = obj_dir.parent.parent / "trainings"
  written = []
  header = (
    f"# Dexonomy {obj_name} grasps ({len(rows)}, {len(types)} grasp types) from\n"
    "# src/tools/make_dexonomy_grasp_sets.py; grasp sets not in git.\n"
  )

  if args.many_to_many:
    path = trainings / f"{args.name}_m2m.yaml"
    stage = _stage("m2m", obj_name, f"{rel}/all", f"{rel}/all", args.iterations)
    written.append((path, _training(header + "# many-to-many, one stage.\n", [stage])))

  if args.many_to_one:
    pose = pool["object_pose"][rows]
    pos_scale = np.median(
      np.linalg.norm(pose[:, None, :3] - pose[None, :, :3], axis=-1)
    )
    rot_scale = np.median(_rot_dist(pose[:, None, 3:], pose[None, :, 3:]))
    if args.goal_index is None:
      dist = _pose_dist(pose[:, None], pose[None], pos_scale, rot_scale)
      goal = int(dist.mean(1).argmin())  # medoid
    else:
      goal = args.goal_index
    goal_rows = np.array([rows[goal]])
    save_grasp_set_npz(
      root / "goal",
      _grasp_set(pool, goal_rows, tips, args.joint_weights, args.tip_weights),
      HAND,
    )
    others = np.delete(np.arange(len(rows)), goal)
    d = _pose_dist(pose[others], pose[goal], pos_scale, rot_scale)
    order = others[np.argsort(d, kind="stable")]
    k = args.many_to_one
    stages = []
    print(
      f"many-to-one goal: pool row {rows[goal]} ({pool['grasp_type'][rows[goal]]}); "
      f"distance = |dp| / {pos_scale * 1000:.1f} mm + angle / {math.degrees(rot_scale):.0f} deg"
    )
    for s in range(1, k + 1):
      n = round(len(order) * s / k)
      sub = np.sort(order[:n])
      save_grasp_set_npz(
        root / f"init_stage{s}",
        _grasp_set(pool, rows[sub], tips, args.joint_weights, args.tip_weights),
        HAND,
      )
      dp = np.linalg.norm(pose[sub, :3] - pose[goal, :3], axis=-1)
      da = np.degrees(_rot_dist(pose[sub, 3:], pose[goal, 3:]))
      print(
        f"  stage{s}: {n:4d} init, max |dp| {dp.max() * 1000:5.1f} mm, "
        f"max angle {da.max():5.1f} deg (median {np.median(dp) * 1000:.1f} mm, "
        f"{np.median(da):.0f} deg)"
      )
      stages.append(
        _stage(
          f"stage{s}", obj_name, f"{rel}/init_stage{s}", f"{rel}/goal", args.iterations
        )
      )
    path = trainings / f"{args.name}_m2o_curriculum.yaml"
    note = (
      f"# many-to-one curriculum: one goal, {k} stages adding initial grasps by\n"
      "# object pose distance to the goal (nearest first).\n"
    )
    written.append((path, _training(header + note, stages)))

  if args.training:
    for path, text in written:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(text)
      load_training(path)  # validates every stage, incl. joint limits
      print(f"wrote {path}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = ap.add_subparsers(dest="cmd", required=True)
  a = sub.add_parser("synth", help="Dexonomy grasps of all types -> pool npz")
  a.add_argument("--object", default="cube", help="compound object name")
  a.add_argument("--types", nargs="*", help="grasp types (default: the library)")
  a.add_argument("--k", type=int, default=400, help="grasps at most per type and seed")
  a.add_argument("--seeds", type=int, nargs="+", default=[0])
  a.add_argument("--max-rounds", type=int, default=20)
  a.add_argument("--out", required=True)
  a.set_defaults(fn=synth)

  b = sub.add_parser("sets", help="pool(s) -> grasp sets and training files")
  b.add_argument("--pool", nargs="+", required=True)
  b.add_argument("--object-dir", required=True)
  b.add_argument("--name", required=True)
  b.add_argument("--count", type=int, default=3000)
  b.add_argument("--seed", type=int, default=0)
  b.add_argument("--many-to-one", type=int, default=0, metavar="STAGES")
  b.add_argument("--goal-index", type=int, help="many-to-one goal (default: medoid)")
  b.add_argument("--many-to-many", action="store_true")
  b.add_argument("--iterations", type=int, default=1000, help="per stage")
  b.add_argument(
    "--joint-weights",
    choices=("contact", "all"),
    default="contact",
    help="w_joint = 1 for the fingers touching the object, or for all",
  )
  b.add_argument(
    "--tip-weights",
    choices=("pad", "none"),
    default="pad",
    help="w_tip = 1 for the touching pads, or 0 for all",
  )
  b.add_argument("--dedupe-pos", type=float, default=0.005)
  b.add_argument("--dedupe-rot", type=float, default=10.0, help="deg")
  b.add_argument("--training", action="store_true", help="write training files")
  b.set_defaults(fn=sets)
  args = ap.parse_args()
  args.fn(args)


if __name__ == "__main__":
  main()
