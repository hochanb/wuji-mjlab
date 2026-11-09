# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""The sync-free reset methods do what mjlab's do."""

import pytest
import torch
import wuji_mjlab.tasks  # noqa: F401  (installs the patches)
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from wuji_mjlab.rl import sync_free_resets
from wuji_mjlab.rl.sync_free_resets import fill_rows


@pytest.mark.parametrize("dim", [0, 1])
def test_fill_rows_matches_indexing(dim):
  t = torch.randn(6, 6, 2)
  ids = torch.tensor([4, 1])
  expected = t.clone()
  expected[(slice(None),) * dim + (ids,)] = -1.5
  fill_rows(t, ids, -1.5, dim=dim)
  torch.testing.assert_close(t, expected)
  fill_rows(t, slice(None), 0.0)
  assert not t.any()


def test_circular_buffer_reset_matches_mjlab():
  assert sync_free_resets.install()
  a = CircularBuffer(max_len=3, batch_size=5, device="cpu")
  for _ in range(4):
    a.append(torch.randn(5, 2))
  b_buffer, b_pushes = a._buffer.clone(), a._num_pushes.clone()
  ids = torch.tensor([0, 3])
  a.reset(ids)
  b_buffer[:, ids] = 0.0
  b_pushes[ids] = 0
  torch.testing.assert_close(a._buffer, b_buffer)
  torch.testing.assert_close(a._num_pushes, b_pushes)
  a.reset()
  assert not a._buffer.any() and not a._num_pushes.any()


def test_env_resets_zero_rows_and_log_tensors():
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
    example_stage_env_cfg,
  )

  cfg = example_stage_env_cfg()
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.max_common_steps = 10**6
  try:
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(5):
      env.step(act.uniform_(-1, 1))
    rm = env.reward_manager
    sums = {k: v.clone() for k, v in rm._episode_sums.items()}
    ids = torch.tensor([1, 3])
    env.reset(env_ids=ids)
    log = env.extras["log"]
    for k, v in sums.items():
      assert not rm._episode_sums[k][ids].any()
      torch.testing.assert_close(rm._episode_sums[k][[0, 2]], v[[0, 2]])
      torch.testing.assert_close(
        log["Episode_Reward/" + k], v[ids].mean() / env.max_episode_length_s
      )
    for key, value in log.items():
      assert isinstance(value, torch.Tensor | float | int), key
    assert any(k.startswith("Metrics/reorient_command/") for k in log)
  finally:
    env.close()
