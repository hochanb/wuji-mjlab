# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reset paths of mjlab's managers without CPU-GPU synchronisation.

With thousands of envs some env resets nearly every step, and mjlab's reset
code synchronises the CPU with the GPU ~100 times per reset: ``t[ids] = 0.0``
on a CUDA tensor copies the Python scalar to the GPU (a blocking copy), and
command / termination logging calls ``.item()``. Each stall idles the GPU
while the CPU catches up.

``install()`` swaps in equivalent methods that zero rows with
``index_fill_`` (the scalar travels as a kernel argument) and keep logged
values as tensors (rsl-rl's logger takes tensors). Only for the mjlab version
these were written against; another version keeps its own code.
"""

from __future__ import annotations

import importlib.metadata
import warnings

import torch

MJLAB_VERSION = "1.6.0"


def fill_rows(t: torch.Tensor, ids, value, dim: int = 0) -> None:
  """``t[ids] = value`` along ``dim``, without a host-to-device copy when
  ``ids`` is a CUDA index tensor (any other ``ids`` uses plain indexing)."""
  if isinstance(ids, slice) and ids == slice(None):
    t.fill_(value)
  elif (
    isinstance(ids, torch.Tensor)
    and ids.device == t.device
    and ids.dtype in (torch.long, torch.int)
  ):
    t.index_fill_(dim, ids.reshape(-1), value)
  elif dim == 0:
    t[ids] = value
  else:
    t[(slice(None),) * dim + (ids,)] = value


def _circular_buffer_reset(self, batch_ids=None) -> None:
  ids = slice(None) if batch_ids is None else batch_ids
  fill_rows(self._num_pushes, ids, 0)
  if self._buffer is not None:
    fill_rows(self._buffer, ids, 0.0, dim=1)


def _reward_reset(self, env_ids=None):
  if env_ids is None:
    env_ids = slice(None)
  extras = {}
  for key in self._episode_sums.keys():
    episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
    extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
    fill_rows(self._episode_sums[key], env_ids, 0.0)
  for term_cfg in self._class_term_cfgs:
    term_cfg.func.reset(env_ids=env_ids)
  return extras


def _metrics_reset(self, env_ids=None):
  if env_ids is None:
    env_ids = slice(None)
  extras = {}
  counts = self._step_count[env_ids].float()
  safe_counts = torch.clamp(counts, min=1.0)
  for idx, key in enumerate(self._episode_sums):
    reduce = self._term_cfgs[idx].reduce
    if reduce == "max":
      extras["Episode_Metrics/" + key] = torch.mean(self._episode_max[key][env_ids])
      fill_rows(self._episode_max[key], env_ids, float("-inf"))
    elif reduce == "last":
      extras["Episode_Metrics/" + key] = torch.mean(self._step_values[env_ids, idx])
    elif reduce == "sum":
      extras["Episode_Metrics/" + key] = torch.mean(self._episode_sums[key][env_ids])
    else:
      extras["Episode_Metrics/" + key] = torch.mean(
        self._episode_sums[key][env_ids] / safe_counts
      )
    fill_rows(self._episode_sums[key], env_ids, 0.0)
  fill_rows(self._step_count, env_ids, 0)
  for buf in self._substep_accum:
    fill_rows(buf, env_ids, 0.0)
  for term_cfg in self._class_term_cfgs:
    term_cfg.func.reset(env_ids=env_ids)
  return extras


def _termination_reset(self, env_ids=None):
  if env_ids is None:
    env_ids = slice(None)
  extras = {}
  for key in self._term_dones.keys():
    extras["Episode_Termination/" + key] = torch.count_nonzero(
      self._term_dones[key][env_ids]
    )
  for term_cfg in self._class_term_cfgs:
    term_cfg.func.reset(env_ids=env_ids)
  return extras


def _command_term_reset(self, env_ids):
  assert isinstance(env_ids, torch.Tensor)
  extras = {}
  for metric_name, metric_value in self.metrics.items():
    extras[metric_name] = torch.mean(metric_value[env_ids])
    fill_rows(metric_value, env_ids, 0.0)
  fill_rows(self.command_counter, env_ids, 0)
  self._resample(env_ids)
  return extras


_installed = False


def install() -> bool:
  """Swap in the sync-free reset methods (idempotent). False when the
  installed mjlab is not the version they were written against."""
  global _installed
  if _installed:
    return True
  version = importlib.metadata.version("mjlab")
  if version != MJLAB_VERSION:
    warnings.warn(
      f"sync_free_resets: written for mjlab {MJLAB_VERSION}, found {version}; "
      "keeping mjlab's reset code",
      RuntimeWarning,
      stacklevel=2,
    )
    return False
  from mjlab.managers.command_manager import CommandTerm
  from mjlab.managers.metrics_manager import MetricsManager
  from mjlab.managers.reward_manager import RewardManager
  from mjlab.managers.termination_manager import TerminationManager
  from mjlab.utils.buffers.circular_buffer import CircularBuffer

  CircularBuffer.reset = _circular_buffer_reset
  RewardManager.reset = _reward_reset
  MetricsManager.reset = _metrics_reset
  TerminationManager.reset = _termination_reset
  CommandTerm.reset = _command_term_reset
  _installed = True
  return True
