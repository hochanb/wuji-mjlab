# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Play an SGCRL checkpoint in the viser viewer (bound to 127.0.0.1).

Usage:
  pixi run play-sgcrl --task <SGCRL task> --checkpoint-file <model_N.pt> \\
    [--port 8082] [--num-envs 4] [--stochastic] [env.<path>=<value> ...]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
import viser
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import ViserPlayViewer
from wuji_mjlab.rl.sgcrl import SgcrlCfg, SgcrlTrainer
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--port", type=int, default=8082)
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--stochastic", action="store_true", help="sample actions")
  parser.add_argument(
    "--hand-friction",
    type=float,
    default=None,
    help="sliding friction of the hand collision geoms (they win the contact "
    "parameters through their priority); default keeps the model's",
  )
  args, overrides = parser.parse_known_args()

  env_cfg, cfg = prepare_task_cfgs(args.task, overrides, play=True)
  if not isinstance(cfg, SgcrlCfg):
    raise SystemExit(f"{args.task} is not an SGCRL task.")
  env_cfg.scene.num_envs = args.num_envs
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env.max_common_steps = cfg.max_iterations * cfg.steps_per_iteration
  if args.hand_friction is not None:
    hand = torch.as_tensor(
      (env.sim.mj_model.geom_priority > 0).nonzero()[0], device=device
    )
    env.sim.model.geom_friction[..., hand, 0] = args.hand_friction
  agent = SgcrlTrainer(env, replace(cfg, buffer_steps=2), None, device)
  agent.load(args.checkpoint_file, actor_only=True)
  wrapped = RslRlVecEnvWrapper(env)

  def policy(obs):
    return agent.act(obs, deterministic=not args.stochastic)

  ViserPlayViewer(
    wrapped, policy, viser_server=viser.ViserServer(host="127.0.0.1", port=args.port)
  ).run()


if __name__ == "__main__":
  main()
