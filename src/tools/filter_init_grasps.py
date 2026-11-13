"""Drop the initial grasps of a training stage that do not survive the
action warmup (the fingers hold the reset pose, no policy): each grasp is
reset ``--trials`` times in the stage's env (domain randomization on) and
fails a trial if, by the end of the warmup, the object left the cage, hit
the speed limit, or moved more than ``--max-move`` from where it started in
the palm frame. Grasps failing more than ``--max-fails`` trials go; the rest
are saved as ``<grasps>/<out>/`` next to the source set (not in git).

usage (repo root):
  pixi run python src/tools/filter_init_grasps.py --training <training.yaml> \\
    [--stage 0] --out cube_dex_3k_stable [--trials 4] [--max-move 0.02]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import build_stage_env_cfg
from wuji_mjlab.tasks.reorient.grasp_stage import mdp as gmdp
from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
  GraspSet,
  load_grasp_sets,
  load_training,
  save_grasp_set_npz,
)
from wuji_mjlab.tasks.reorient.mdp.actions import warmup_steps
from wuji_mjlab.tasks.reorient.mdp.cage import read_cage_penalty_counter


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--training", required=True)
  ap.add_argument("--stage", type=int, default=0)
  ap.add_argument("--out", required=True, help="output folder name under grasps/")
  ap.add_argument("--trials", type=int, default=4)
  ap.add_argument("--max-fails", type=int, default=0)
  ap.add_argument("--max-move", type=float, default=0.02, help="[m] in the warmup")
  ap.add_argument("--num-envs", type=int, default=4096)
  a = ap.parse_args()
  os.environ.pop("DISPLAY", None)

  training = load_training(a.training)
  stage = training.stages[a.stage]
  (obj,) = stage.objects
  folders = stage.init_set_folders[obj.name]
  grasps = load_grasp_sets(folders, training.hand)
  n = len(grasps)
  cfg = build_stage_env_cfg(training, a.stage)
  cfg.scene.num_envs = a.num_envs
  env = ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  env.max_common_steps = 10**9
  act_cfg = cfg.actions["finger_joint_pos"]
  steps = warmup_steps(act_cfg.warmup_time_s, env.step_dt)
  robot, objent = env.scene["robot"], env.scene["object"]
  palm = robot.find_bodies(".*_palm_link")[0][0]
  tm = env.termination_manager

  # The reset event draws its rows from sample_rows: hand it ours instead.
  queue: list[torch.Tensor] = []
  orig = gmdp.sample_rows

  def chosen_rows(tables, count, device):
    idx = (
      queue.pop(0)
      if queue and len(queue[0]) == count
      else torch.zeros(count, dtype=torch.long)
    )
    return idx, gmdp.gather_rows(tables, idx, device)

  gmdp.sample_rows = chosen_rows
  order = np.tile(np.arange(n), a.trials)
  fails = np.zeros(n, dtype=int)
  moves, reasons = [], {"cage": 0, "speed": 0, "move": 0}
  zero = torch.zeros(a.num_envs, env.action_manager.total_action_dim, device=env.device)

  def palm_frame():
    rel = objent.data.root_link_pos_w - robot.data.body_link_pos_w[:, palm]
    return quat_apply_inverse(robot.data.body_link_quat_w[:, palm], rel)

  with torch.inference_mode():
    for start in range(0, len(order), a.num_envs):
      batch = order[start : start + a.num_envs]
      idx = np.resize(batch, a.num_envs)  # pad the last batch
      queue[:] = [torch.as_tensor(idx, dtype=torch.long)]
      env.reset()
      p0 = palm_frame().clone()
      speed = torch.zeros(a.num_envs, dtype=torch.bool, device=env.device)
      for _ in range(steps):
        env.step(zero)
        speed |= tm.get_term("object_speed_limit").bool()
      counter = read_cage_penalty_counter(env)
      cage = (counter > 0) if counter is not None else torch.zeros_like(speed)
      move = (palm_frame() - p0).norm(dim=-1)
      bad = cage | speed | (move > a.max_move)
      k = len(batch)
      np.add.at(fails, batch, bad[:k].cpu().numpy().astype(int))
      moves.append(move[:k].cpu().numpy())
      reasons["cage"] += int(cage[:k].sum())
      reasons["speed"] += int(speed[:k].sum())
      reasons["move"] += int((move[:k] > a.max_move).sum())
  gmdp.sample_rows = orig

  moves = np.concatenate(moves) * 1000
  keep = fails <= a.max_fails
  print(
    f"{n} grasps x {a.trials} trials, warmup {steps} steps: move in warmup "
    f"median {np.median(moves):.1f} mm, p90 {np.percentile(moves, 90):.1f} mm; "
    f"trials failing: out of cage {reasons['cage']}, speed limit {reasons['speed']}, "
    f"moved > {a.max_move * 1000:.0f} mm {reasons['move']} (of {len(order)})"
  )
  print(
    "grasps by failed trials: "
    + ", ".join(f"{f}: {int((fails == f).sum())}" for f in range(a.trials + 1))
  )
  rows = np.flatnonzero(keep)
  out = Path(folders[0]).parent.parent / a.out / "all"
  subset = GraspSet(
    grasps.ids[rows], *(getattr(grasps, f)[rows] for f in GraspSet.ARRAYS)
  )
  save_grasp_set_npz(out, subset, training.hand)
  print(f"kept {len(rows)} / {n} -> {out}")


if __name__ == "__main__":
  main()
