# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import pytest
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.parametrize("hold", [False, True])
def test_warmup_target_follows_option(hold):
  cfg = load_env_cfg("WujiHand2_ReorientStaplerFixedGoalWrist_50Hz", play=True)
  cfg.scene.num_envs = 2
  cfg.actions["finger_joint_pos"].warmup_hold_reset_pose = hold
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  term = env.action_manager.get_term("finger_joint_pos")
  reset_pose = env.scene["robot"].data.joint_pos[:, term._target_ids].clone()
  env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
  expected = reset_pose if hold else term._default_joint_pos
  torch.testing.assert_close(term.processed_action, expected)
  # The reset event opens the fingers off the cage pose, so the two differ.
  assert not torch.allclose(reset_pose, term._default_joint_pos)
  env.close()
