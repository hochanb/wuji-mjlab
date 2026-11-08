# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Train a grasp-stage task with PPO through the stages of a training file.

Each stage gets a freshly built env (``env.close()`` then a new env for the
next stage's objects and grasps); the policy, optimiser and iteration count
carry over, and a checkpoint is written at every stage switch. All stages
are validated before the first env is built.

With ``--gpu-ids 0,1`` one worker per GPU trains a single policy (rsl_rl
multi-GPU, like ``train_rsl_rl.py``); ``env.scene.num_envs`` is per GPU.
The wrist is fixed at the mount unless ``--wrist`` adds the actuated wrist.

Usage:
  CUDA_VISIBLE_DEVICES=0 pixi run train-stages \\
    [--training <training.yaml>] [--gpu-ids 0,1] [--wrist] \\
    [env.<path>=<value> ...] [agent.<path>=<value> ...]
"""

from __future__ import annotations

import argparse
import hashlib
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
from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
  grasp_set_file,
  load_grasp_set,
  load_training,
)
from wuji_mjlab.utils.override_utils import apply_overrides

TASK = "WujiHand2_GraspStage_50Hz"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _train(
  training_path: str,
  agent_cfg,
  env_overrides: list[str],
  log_dir: Path,
  wrist: bool,
) -> None:
  """Run every stage on this process's device (one call per GPU worker)."""
  training = load_training(training_path)
  if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
  else:
    local_rank, device = 0, "cpu"
  rank = int(os.environ.get("RANK", "0"))
  agent_cfg.seed += local_rank  # distinct env randomness per worker
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  runner, common_steps = None, 0
  for i, stage in enumerate(training.stages):
    env_cfg = build_stage_env_cfg(training, i, wrist=wrist)
    apply_overrides({"env": env_cfg}, env_overrides)
    env_cfg.seed = agent_cfg.seed + i
    if rank == 0:
      dump_yaml(log_dir / "params" / f"env_stage{i}_{stage.name}.yaml", asdict(env_cfg))
      print(
        f"[INFO] stage {i} '{stage.name}': {stage.advance_after_iterations} iterations"
      )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if rank == 0:
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
    if rank == 0:
      runner.save(
        str(
          log_dir
          / f"model_{runner.current_learning_iteration}_stage{i}_{stage.name}.pt"
        )
      )
    runner.current_learning_iteration += 1  # the next stage starts a new iteration
    common_steps = env.common_step_counter
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--training", default=str(EXAMPLE_TRAINING))
  parser.add_argument(
    "--gpu-ids",
    default=None,
    help="comma-separated GPU ids for multi-GPU training, e.g. 0,1 "
    "(default: one process on the visible device)",
  )
  parser.add_argument(
    "--wrist",
    action="store_true",
    help="train with the actuated wrist (default: wrist fixed at the mount)",
  )
  args, overrides = parser.parse_known_args()
  env_overrides = [o for o in overrides if o.startswith("env.")]
  agent_overrides = [o for o in overrides if o.startswith("agent.")]
  if len(env_overrides) + len(agent_overrides) != len(overrides):
    raise SystemExit(f"Overrides must start with env. or agent.: {overrides}")
  gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None
  wrist = args.wrist

  training = load_training(args.training)  # validates every stage up front
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.logger = "tensorboard"
  agent_cfg.upload_model = False
  apply_overrides({"agent": agent_cfg}, agent_overrides)
  agent_cfg.max_iterations = training.total_iterations

  name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    name += f"_{agent_cfg.run_name}"
  log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / name
  (log_dir / "params").mkdir(parents=True, exist_ok=True)
  shutil.copy(training.path, log_dir / "params" / "training.yaml")
  # Object files are copied; grasp sets (possibly millions of grasps) are
  # recorded by path, checksum and size instead.
  records = []
  for stage in training.stages:
    for obj in stage.objects:
      dst = log_dir / "params" / "objects" / obj.name
      dst.mkdir(parents=True, exist_ok=True)
      shutil.copy(obj.folder / "object.yaml", dst / "object.yaml")
      for role, folders in (
        ("init", stage.init_set_folders[obj.name]),
        ("goal", stage.goal_set_folders[obj.name]),
      ):
        for folder in folders:
          path = grasp_set_file(folder)
          records.append(
            {
              "stage": stage.name,
              "object": obj.name,
              "role": role,
              "file": str(path),
              "sha256": _sha256(path),
              "bytes": path.stat().st_size,
              "grasps": len(load_grasp_set(folder, training.hand)),
            }
          )
  dump_yaml(log_dir / "params" / "grasp_sets.yaml", {"grasp_sets": records})
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  dump_yaml(
    log_dir / "params" / "launch.yaml",
    {"wrist": wrist, "gpu_ids": gpu_ids, "env_overrides": env_overrides},
  )
  print(
    f"[INFO] {len(training.stages)} stage(s), {training.total_iterations} "
    f"iterations, wrist {'actuated' if wrist else 'fixed'}, "
    f"GPUs {gpu_ids or 'default'}, logging to {log_dir}"
  )

  train_args = (str(training.path), agent_cfg, env_overrides, log_dir, wrist)
  if gpu_ids is None or len(gpu_ids) == 1:
    if gpu_ids:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    _train(*train_args)
    return

  import logging

  import torchrunx

  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
  os.environ["MUJOCO_GL"] = "egl"
  os.environ.setdefault("TORCHRUNX_LOG_DIR", str(log_dir / "torchrunx"))
  logging.basicConfig(level=logging.INFO)
  print(f"[INFO] Launching training with {len(gpu_ids)} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=len(gpu_ids),
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
    + ("MUJOCO*", "WUJI_ACTION_DIAGNOSTICS", "WUJI_SIM_CHUNK_SIZE"),
  ).run(_train, *train_args)


if __name__ == "__main__":
  main()
