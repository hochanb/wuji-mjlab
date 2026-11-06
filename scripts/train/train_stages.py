# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Train a grasp-stage task with PPO through the stages of a training file.

Each stage gets a freshly built env (``env.close()`` then a new env for the
next stage's objects and grasps); the policy, optimiser and iteration count
carry over, and a checkpoint is written at every stage switch. All stages
are validated before the first env is built.

Usage:
  CUDA_VISIBLE_DEVICES=0 pixi run train-stages \\
    [--training <training.yaml>] [env.<path>=<value> ...] [agent.<path>=<value> ...]
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
  EXAMPLE_TRAINING,
  build_stage_env_cfg,
)
from wuji_mjlab.tasks.reorient.grasp_stage.mdp import observation_layout
from wuji_mjlab.tasks.reorient.grasp_stage.schema import load_training
from wuji_mjlab.utils.override_utils import apply_overrides

TASK = "WujiHand2_GraspStage_50Hz"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--training", default=str(EXAMPLE_TRAINING))
  args, overrides = parser.parse_known_args()
  env_overrides = [o for o in overrides if o.startswith("env.")]
  agent_overrides = [o for o in overrides if o.startswith("agent.")]
  if len(env_overrides) + len(agent_overrides) != len(overrides):
    raise SystemExit(f"Overrides must start with env. or agent.: {overrides}")

  training = load_training(args.training)  # validates every stage up front
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.logger = "tensorboard"
  agent_cfg.upload_model = False
  apply_overrides({"agent": agent_cfg}, agent_overrides)
  agent_cfg.max_iterations = training.total_iterations

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()
  name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    name += f"_{agent_cfg.run_name}"
  log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / name
  (log_dir / "params").mkdir(parents=True, exist_ok=True)
  shutil.copy(training.path, log_dir / "params" / "training.yaml")
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  print(
    f"[INFO] {len(training.stages)} stage(s), {training.total_iterations} "
    f"iterations, device {device}, logging to {log_dir}"
  )

  runner, common_steps = None, 0
  for i, stage in enumerate(training.stages):
    env_cfg = build_stage_env_cfg(training, i)
    apply_overrides({"env": env_cfg}, env_overrides)
    env_cfg.seed = agent_cfg.seed + i
    dump_yaml(log_dir / "params" / f"env_stage{i}_{stage.name}.yaml", asdict(env_cfg))
    print(
      f"[INFO] stage {i} '{stage.name}': {stage.advance_after_iterations} iterations"
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    dump_yaml(
      log_dir / "params" / f"obs_layout_stage{i}_{stage.name}.yaml",
      observation_layout(env),
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if runner is None:
      runner = WujiOnPolicyRunner(wrapped, asdict(agent_cfg), str(log_dir), device)
      runner.add_git_repo_to_log(__file__)
    else:
      # Same env count and observation layout: keep the policy and storage.
      runner.env = wrapped
      env.max_common_steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env
      env.common_step_counter = common_steps  # time-ramped terms carry on
    runner.learn(stage.advance_after_iterations, init_at_random_ep_len=True)
    runner.save(
      str(
        log_dir / f"model_{runner.current_learning_iteration}_stage{i}_{stage.name}.pt"
      )
    )
    runner.current_learning_iteration += 1  # the next stage starts a new iteration
    common_steps = env.common_step_counter
    env.close()


if __name__ == "__main__":
  main()
