"""Goal-set curriculum over grasp types: nested goal sets of growing size
(e.g. 1, 3, 10, 30) drawn from a grasp set, one grasp per grasp type first.

Dexonomy-dependent: needs the set's per-grasp ``grasp_type`` metadata
(``schema.load_grasp_meta``), which make_dexonomy_grasp_sets writes; sets
from generators without grasp types do not qualify.

From ``<grasps>/<src>/all`` and its grasp types:

1. one representative per grasp type: the type's medoid under the distance
   below (``--first`` replaces its type's representative);
2. more grasps, if the largest size asks for more than there are types,
   each the one farthest from those chosen so far;
3. an order: ``--first`` (or the overall medoid), then farthest-point
   sampling, so every prefix is spread out; the sets are its prefixes.

Distance between grasps: |dp| / pos_scale + angle / rot_scale + mean |dq| /
q_scale, each scale the median pairwise value. Writes
``<grasps>/<name>/goals_<k>`` (with their metadata) and
``<trainings>/<name>_curriculum.yaml`` with ``all`` as the initial grasps in
every stage (not in git: the sets).

Usage (repo root):
  pixi run python src/tools/make_goal_type_curriculum.py \\
    --object-dir src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube \\
    --src cube_dex_3k --name cube_dex_3k_goals30 --sizes 1 3 10 30 \\
    --first 3730 --iterations 1250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
  GraspSet,
  load_grasp_meta,
  load_grasp_set,
  save_grasp_meta,
  save_grasp_set_npz,
)

HAND = "wuji_hand2_right"


def _rot_dist(qa, qb):
  dot = np.abs((np.asarray(qa) * np.asarray(qb)).sum(-1))
  return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _distances(s: GraspSet) -> np.ndarray:
  pos, quat = s.object_pos.astype(np.float64), s.object_quat.astype(np.float64)
  q = s.joint_q.astype(np.float64)
  dp = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
  da = _rot_dist(quat[:, None], quat[None])
  dq = np.abs(q[:, None] - q[None]).mean(-1)
  off = ~np.eye(len(s), dtype=bool)
  return dp / np.median(dp[off]) + da / np.median(da[off]) + dq / np.median(dq[off])


def _subset(s: GraspSet, rows: np.ndarray) -> GraspSet:
  return GraspSet(s.ids[rows], *(getattr(s, a)[rows] for a in GraspSet.ARRAYS))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--object-dir", required=True)
  ap.add_argument("--src", required=True, help="grasp sets folder under grasps/")
  ap.add_argument("--name", required=True)
  ap.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 10, 30])
  ap.add_argument("--first", type=int, default=None, help="grasp id of the first goal")
  ap.add_argument("--iterations", type=int, default=1250)
  a = ap.parse_args()

  obj_dir = Path(a.object_dir)
  src = obj_dir / "grasps" / a.src
  grasps = load_grasp_set(src / "all", HAND)
  meta = load_grasp_meta(src / "all", len(grasps))
  if meta is None or "grasp_type" not in meta.rows:
    raise SystemExit(
      f"{src / 'all'}: no per-grasp grasp_type metadata (Dexonomy sets from "
      "make_dexonomy_grasp_sets.py have it)"
    )
  types = meta["grasp_type"].astype(str)
  dist = _distances(grasps)

  reps = {}
  for t in sorted(set(types)):
    rows = np.flatnonzero(types == t)
    reps[t] = int(rows[dist[np.ix_(rows, rows)].mean(1).argmin()])
  if a.first is not None:
    first = int(np.flatnonzero(grasps.ids == a.first)[0])
    reps[types[first]] = first
  else:
    first = int(dist.mean(1).argmin())
  chosen = list(reps.values())
  n = max(a.sizes)
  while len(chosen) < n:  # more than one per type: the farthest ones
    rest = np.setdiff1d(np.arange(len(grasps)), chosen)
    chosen.append(int(rest[dist[np.ix_(rest, chosen)].min(1).argmax()]))
  chosen = np.array(chosen)
  # Order: the first goal, then farthest-point sampling among the chosen.
  order = [first]
  while len(order) < len(chosen):
    rest = np.setdiff1d(chosen, order)
    order.append(int(rest[dist[np.ix_(rest, order)].min(1).argmax()]))
  order = np.array(order[:n])

  rel = obj_dir.name
  stages = []
  for k in a.sizes:
    rows = order[:k]
    out = obj_dir / "grasps" / a.name / f"goals_{k}"
    save_grasp_set_npz(out, _subset(grasps, rows), HAND)
    save_grasp_meta(out, meta.subset(rows), k)
    kinds = sorted({types[r] for r in rows})
    print(
      f"goals_{k}: {k} grasps, {len(kinds)} types: {', '.join(kinds) if k <= 10 else '...'}"
    )
    stages.append(
      f"  - name: goals{k}\n"
      f"    objects: [{rel}]\n"
      f"    init_grasps: {{{rel}: [grasps/{a.src}/all]}}\n"
      f"    goal_grasps: {{{rel}: [grasps/{a.name}/goals_{k}]}}\n"
      f"    advance: {{after_iterations: {a.iterations}}}\n"
    )
  print(
    f"pairwise distance among the {n}: min "
    f"{dist[np.ix_(order, order)][~np.eye(n, dtype=bool)].min():.2f} "
    f"(median between any two grasps: 3 by construction)"
  )
  trainings = obj_dir.parent.parent / "trainings"
  path = trainings / f"{a.name}_curriculum.yaml"
  path.write_text(
    f"# Many-to-many goal curriculum: all {len(grasps)} {a.src} grasps as initial\n"
    f"# states; goal sets of {', '.join(map(str, a.sizes))} grasps (one per grasp type\n"
    "# first, spread out by farthest-point sampling; nested), from\n"
    "# src/tools/make_goal_type_curriculum.py; grasp sets not in git.\n"
    "version: 1\n"
    f"hand: {HAND}\n"
    "encoding: {name: none, dim: 0}\n"
    "object_root: ../objects\n"
    "stages:\n" + "".join(stages)
  )
  print("wrote", path)


if __name__ == "__main__":
  main()
