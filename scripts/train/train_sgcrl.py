# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Train a task with single-goal contrastive RL (SGCRL).

Usage:
  CUDA_VISIBLE_DEVICES=0 pixi run train-sgcrl --task <SGCRL task> \\
    [env.<path>=<value> ...] [agent.<path>=<value> ...]

``agent.*`` overrides set ``SgcrlCfg`` fields (e.g. ``agent.batch_size=4096``).
Logs and checkpoints go to logs/sgcrl/<experiment_name>/<date>_<run_name>.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.sgcrl import SgcrlCfg, SgcrlTrainer
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--task", required=True)
  parser.add_argument(
    "--resume",
    default=None,
    help="checkpoint to continue from, in its run directory (pass the same overrides)",
  )
  args, overrides = parser.parse_known_args()

  env_cfg, cfg = prepare_task_cfgs(args.task, overrides, play=False)
  if not isinstance(cfg, SgcrlCfg):
    raise SystemExit(f"{args.task} is not an SGCRL task (rl cfg {type(cfg).__name__}).")

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  env_cfg.seed = cfg.seed

  if args.resume:
    # Continue in the checkpoint's run directory (its params stay as written).
    log_dir = Path(args.resume).parent
  else:
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.run_name:
      name += f"_{cfg.run_name}"
    log_dir = Path("logs") / "sgcrl" / cfg.experiment_name / name
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg))
  print(f"[INFO] SGCRL on {args.task}, device {device}, logging to {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  # Time-ramped terms (e.g. the object disturbance) read the total step count.
  env.max_common_steps = (
    cfg.ramp_iterations or cfg.max_iterations
  ) * cfg.steps_per_iteration
  trainer = SgcrlTrainer(env, cfg, str(log_dir), device)
  if args.resume:
    trainer.load(args.resume)
    env.common_step_counter = trainer.iteration * cfg.steps_per_iteration
  trainer.learn()
  env.close()


if __name__ == "__main__":
  main()
