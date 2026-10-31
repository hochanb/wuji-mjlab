# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Single-goal contrastive RL (SGCRL) configuration.

Liu, Tang, Eysenbach, "A Single Goal is All You Need" (arXiv:2408.05804):
contrastive RL whose data is always collected toward one fixed goal. Defaults
follow the paper's Table 2 where it applies; the data/update ratio is set for
thousands of GPU environments instead of the paper's four actors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class SgcrlCfg:
  experiment_name: str = "sgcrl"
  run_name: str = ""
  seed: int = 0
  # Observation groups: goal-free state, achieved goal (a function of the
  # state), and the single commanded goal in the same layout.
  state_group: str = "crl_state"
  goal_group: str = "crl_goal"
  target_group: str = "crl_target"

  # Training length and data/update ratio.
  max_iterations: int = 20_000
  # Iterations the env's time-ramped terms (e.g. the object disturbance) ramp
  # over; None uses max_iterations. Keep the original value when extending a
  # finished run so the ramp does not restart partway.
  ramp_iterations: int | None = None
  steps_per_iteration: int = 8  # env steps collected per iteration
  updates_per_iteration: int = 16  # gradient steps per iteration
  batch_size: int = 2048
  random_action_steps: int = 50  # env steps of uniform random actions first
  buffer_steps: int = 1000  # trajectory buffer length per env [steps]

  # Contrastive RL.
  discount: float = 0.99
  # A future goal past an ended episode is its last state (absorbing, e.g. the
  # dropped object); False truncates the offset to the episode instead.
  absorb_episode_end: bool = True
  repr_dim: int = 64
  hidden_dims: tuple[int, ...] = (512, 512)
  critic_lr: float = 3e-4
  actor_lr: float = 3e-4
  alpha_lr: float = 3e-4
  logsumexp_coef: float = 0.01  # LogSumExp regulariser on the InfoNCE logits
  target_entropy: float = 0.0
  actor_min_std: float = 1e-6
  # Fraction of actor goals taken from the batch's own future states; the rest
  # are shuffled across the batch (the paper's random_goals = 0.5).
  actor_own_goal_frac: float = 0.5
  normalize_obs: bool = True

  # Logging and checkpoints.
  log_interval: int = 10
  save_interval: int = 500
