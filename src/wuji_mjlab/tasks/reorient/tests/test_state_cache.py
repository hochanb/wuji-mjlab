# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Cached per-state values match recomputing them, across steps, resets and
goal changes."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
  example_stage_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp import state_cache
from wuji_mjlab.tasks.reorient.mdp.observations import tag_pose_w


def test_cached_values_match_fresh_ones():
  cfg = example_stage_env_cfg()
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.max_common_steps = 10**6
  try:
    cmd = env.command_manager.get_term("reorient_command")
    robot = cmd.robot
    assert state_cache.epoch(robot) is not None  # installed by the command
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for step in range(12):
      act.uniform_(-1.0, 1.0)
      env.step(act)
      if step == 5:  # resets and a new goal in between
        env.reset(env_ids=torch.tensor([1, 2]))
      for _ in range(2):  # the second read comes from the cache
        sites = robot.data.site_pose_w
        tag = sites[:, cmd.tag_site_id]
        pos, quat = tag_pose_w(robot, [cmd.tag_site_id])
        torch.testing.assert_close(pos, tag[:, :3])
        torch.testing.assert_close(quat, tag[:, 3:7])
        torch.testing.assert_close(
          cmd.goal_quat_w, quat_mul(tag[:, 3:7], cmd.goal_quat_tag)
        )
        torch.testing.assert_close(
          cmd.goal_pos_w, tag[:, :3] + quat_apply(tag[:, 3:7], cmd.goal_pos_tag)
        )
        tips = sites[:, cmd._tip_site_ids]
        offset = cmd._tip_point_offset.expand(env.num_envs, -1, -1)
        torch.testing.assert_close(
          cmd.fingertip_pos_w, tips[..., :3] + quat_apply(tips[..., 3:7], offset)
        )
  finally:
    env.close()
