"""Grasp-stage training files from one grasp set, whatever made it.

From ``<grasps>/<src>/all`` (a grasp set; per-grasp metadata, if any,
follows the subsets):

- ``--many-to-many``: init = goal = all, one stage
  (``trainings/<src>_m2m.yaml``);
- ``--many-to-one K``: one goal grasp (``<src>/goal``; the medoid of the
  object poses, or ``--goal-index``) and the other grasps as initial states
  in K cumulative stages ``<src>/init_stage1..K``, nearest object pose to the
  goal first (``trainings/<src>_m2o_curriculum.yaml``).

Object pose distance: |dp| / pos_scale + angle / rot_scale, each scale the
median pairwise value. Only the grasp set is read, so any grasp generator
works (Dexonomy: src/tools/dexonomy/make_dexonomy_grasp_sets.py). The sets
are written; the training files only with ``--write`` (validated by
loading them). Grasp sets are not in git.

Usage (repo root):
  OBJ=src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube
  pixi run python src/tools/make_grasp_trainings.py --object-dir $OBJ \\
    --src cube_dex_3k --many-to-one 5 --many-to-many --write
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
  GraspSet,
  load_grasp_meta,
  load_grasp_set,
  load_training,
  save_grasp_meta,
  save_grasp_set_npz,
)

HAND = "wuji_hand2_right"


def _rot_dist(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
  dot = np.abs((np.asarray(qa) * np.asarray(qb)).sum(-1))
  return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _pose_dist(pos, quat, ref_pos, ref_quat, pos_scale, rot_scale):
  return (
    np.linalg.norm(pos - ref_pos, axis=-1) / pos_scale
    + _rot_dist(quat, ref_quat) / rot_scale
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


def _save(folder: Path, grasps: GraspSet, meta, rows: np.ndarray) -> None:
  save_grasp_set_npz(
    folder,
    GraspSet(grasps.ids[rows], *(getattr(grasps, a)[rows] for a in GraspSet.ARRAYS)),
    HAND,
  )
  if meta is not None:
    save_grasp_meta(folder, meta.subset(rows), len(rows))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--object-dir", required=True)
  ap.add_argument(
    "--src", required=True, help="grasp sets folder under grasps/ (has all/)"
  )
  ap.add_argument("--many-to-one", type=int, default=0, metavar="STAGES")
  ap.add_argument(
    "--goal-index", type=int, help="many-to-one goal row (default: medoid)"
  )
  ap.add_argument("--many-to-many", action="store_true")
  ap.add_argument("--iterations", type=int, default=1000, help="per stage")
  ap.add_argument("--write", action="store_true", help="write the training files")
  a = ap.parse_args()

  obj_dir = Path(a.object_dir)
  obj_name = obj_dir.name
  root = obj_dir / "grasps" / a.src
  rel = f"grasps/{a.src}"
  grasps = load_grasp_set(root / "all", HAND)
  meta = load_grasp_meta(root / "all", len(grasps))
  n = len(grasps)
  kinds = ""
  if meta is not None and "grasp_type" in meta.rows:
    kinds = f", {len(np.unique(meta['grasp_type']))} grasp types"
  source = f" ({meta.source})" if meta is not None and meta.source else ""
  header = (
    f"# {obj_name} grasps {rel}{source}: {n}{kinds}; training from\n"
    "# src/tools/make_grasp_trainings.py; grasp sets not in git.\n"
  )
  trainings = obj_dir.parent.parent / "trainings"
  written = []

  if a.many_to_many:
    path = trainings / f"{a.src}_m2m.yaml"
    stage = _stage("m2m", obj_name, f"{rel}/all", f"{rel}/all", a.iterations)
    written.append((path, _training(header + "# many-to-many, one stage.\n", [stage])))

  if a.many_to_one:
    pos, quat = (
      grasps.object_pos.astype(np.float64),
      grasps.object_quat.astype(np.float64),
    )
    pos_scale = np.median(np.linalg.norm(pos[:, None] - pos[None], axis=-1))
    rot_scale = np.median(_rot_dist(quat[:, None], quat[None]))
    if a.goal_index is None:
      dist = _pose_dist(
        pos[:, None], quat[:, None], pos[None], quat[None], pos_scale, rot_scale
      )
      goal = int(dist.mean(1).argmin())  # medoid
    else:
      goal = a.goal_index
    _save(root / "goal", grasps, meta, np.array([goal]))
    others = np.delete(np.arange(n), goal)
    d = _pose_dist(
      pos[others], quat[others], pos[goal], quat[goal], pos_scale, rot_scale
    )
    order = others[np.argsort(d, kind="stable")]
    k = a.many_to_one
    kind = f" ({meta['grasp_type'][goal]})" if kinds else ""
    print(
      f"many-to-one goal: row {goal}, id {grasps.ids[goal]}{kind}; distance = |dp| / "
      f"{pos_scale * 1000:.1f} mm + angle / {math.degrees(rot_scale):.0f} deg"
    )
    stages = []
    for s in range(1, k + 1):
      sub = np.sort(order[: round(len(order) * s / k)])
      _save(root / f"init_stage{s}", grasps, meta, sub)
      dp = np.linalg.norm(pos[sub] - pos[goal], axis=-1)
      da = np.degrees(_rot_dist(quat[sub], quat[goal]))
      print(
        f"  stage{s}: {len(sub):4d} init, max |dp| {dp.max() * 1000:5.1f} mm, "
        f"max angle {da.max():5.1f} deg (median {np.median(dp) * 1000:.1f} mm, "
        f"{np.median(da):.0f} deg)"
      )
      stages.append(
        _stage(
          f"stage{s}", obj_name, f"{rel}/init_stage{s}", f"{rel}/goal", a.iterations
        )
      )
    note = (
      f"# many-to-one curriculum: one goal, {k} stages adding initial grasps by\n"
      "# object pose distance to the goal (nearest first).\n"
    )
    written.append(
      (trainings / f"{a.src}_m2o_curriculum.yaml", _training(header + note, stages))
    )

  if a.write:
    for path, text in written:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(text)
      load_training(path)  # validates every stage, incl. joint limits
      print("wrote", path)


if __name__ == "__main__":
  main()
