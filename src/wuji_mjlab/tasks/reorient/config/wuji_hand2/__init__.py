# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 reorient task registration (right hand only)."""

from dataclasses import replace

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient import mdp

from .env_cfgs import wuji_hand2_reorient_env_cfg
from .rsl_rl.ppo import wuji_hand2_reorient_ppo_runner_cfg


def _hand2_rl_cfg(run_name: str, max_iterations: int):
  cfg = wuji_hand2_reorient_ppo_runner_cfg(
    run_name=run_name, max_iterations=max_iterations
  )
  return replace(cfg, experiment_name="wuji_reorient_hand2")


register_mjlab_task(
  task_id="WujiHand2_Reorient",
  env_cfg=wuji_hand2_reorient_env_cfg(num_envs=8192),
  play_env_cfg=wuji_hand2_reorient_env_cfg(play=True),
  rl_cfg=_hand2_rl_cfg(run_name="Reorient_Hand2", max_iterations=5000),
  runner_cls=WujiOnPolicyRunner,
)


_RATE_RATIO = 2.5  # 50 Hz / 20 Hz control steps


def _reorient_hand2_50hz_env_cfg(num_envs: int = 8192, play: bool = False):
  """50 Hz deployment recipe for right-hand operation.

  The hold / goal-switch / cage / episode window knobs are rescaled together
  (×2.5) to preserve their wall-clock duration; retune them together, not
  knob-by-knob.
  """
  cfg = wuji_hand2_reorient_env_cfg(hand_side="right", num_envs=num_envs, play=play)
  cfg.decimation = 2
  cfg.actions["joint_pos"].ema_alpha = 0.3
  cfg.rewards["action_rate"].weight = 0.0
  cfg.rewards["action_hf"].weight = -30.0
  cfg.commands["reorient_command"].success_hold_steps = 13  # 5 × 2.5
  cfg.commands["reorient_command"].goal_switch_delay = 50  # 20 × 2.5
  # per-step-count reward — ÷2.43 to hold the per-cycle integral (Σ1..N ∝ N²)
  cfg.rewards["hold_escalation"].weight = 4.694  # 11.4 × (210/1275) × 2.5
  # Curriculum is cleared in play/eval mode, so guard the key.
  if "adaptive_episode" in cfg.curriculum:
    cfg.curriculum["adaptive_episode"].params["target_steps"] = 2000  # 800 × 2.5
  cfg.rewards["cage_escape"].params["max_outside_steps"] = 25  # 10 × 2.5
  cfg.rewards["cage_escape"].params["max_count"] = 37.5  # 1.5 × 25
  cfg.terminations["cage_drop"].params["max_outside_steps"] = 25
  cfg.observations["critic"].terms["cage_counter"].params["max_outside_steps"] = 25
  return cfg


def _reorient_hand2_50hz_rl_cfg():
  cfg = _hand2_rl_cfg(run_name="Reorient_Hand2_50Hz", max_iterations=5000)
  cfg.num_steps_per_env = 100  # 40 × 2.5 → same 2 s rollout horizon
  cfg.algorithm.gamma = 0.996  # 0.99 ** 0.4 → same real-time discount horizon
  return cfg


register_mjlab_task(
  task_id="WujiHand2_Reorient_50Hz",
  env_cfg=_reorient_hand2_50hz_env_cfg(num_envs=8192),
  play_env_cfg=_reorient_hand2_50hz_env_cfg(play=True),
  rl_cfg=_reorient_hand2_50hz_rl_cfg(),
  runner_cls=WujiOnPolicyRunner,
)


# Goal-position box in the wrist-tag frame [m]: per-axis 10-90th percentile of
# the cube position reached by the orientation-only 50 Hz policy (model_4999).
_GOAL_POS_RANGE = ((-0.179, -0.143), (-0.006, 0.019), (-0.006, 0.018))
# Single goal position: per-axis median of the same rollout distribution.
_GOAL_POS_FIXED = ((-0.160, -0.160), (0.008, 0.008), (0.005, 0.005))
_POS_SUCCESS_THRESHOLD = 0.01


def _reorient_pos_hand2_50hz_env_cfg(
  goal_pos_range, num_envs: int = 8192, play: bool = False
):
  """50 Hz recipe plus a position goal: success needs orientation and position."""
  cfg = _reorient_hand2_50hz_env_cfg(num_envs=num_envs, play=play)
  command = cfg.commands["reorient_command"]
  command.goal_pos_range = goal_pos_range
  command.pos_success_threshold = _POS_SUCCESS_THRESHOLD

  policy_terms = cfg.observations["policy"].terms
  critic_terms = cfg.observations["critic"].terms
  history_length = policy_terms["cube_pos_in_tag"].history_length
  params = {"command_name": "reorient_command"}
  policy_terms["goal_pos_err"] = ObservationTermCfg(
    func=mdp.goal_pos_err_in_tag,
    params=params,
    noise=Unoise(n_min=-0.008, n_max=0.008),
    history_length=history_length,
  )
  critic_terms["goal_pos_err"] = ObservationTermCfg(
    func=mdp.goal_pos_err_in_tag, params=params, history_length=history_length
  )
  critic_terms["goal_pos_err_clean"] = ObservationTermCfg(
    func=mdp.goal_pos_err_in_tag, params=params
  )

  cfg.rewards["position_alignment"] = RewardTermCfg(
    func=mdp.position_alignment,
    weight=5.0,  # below orientation_alignment (15)
    params={**params, "bound": 0.01, "margin": 0.03},
  )
  return cfg


def _reorient_pos_hand2_50hz_rl_cfg(experiment_name: str, run_name: str):
  cfg = _reorient_hand2_50hz_rl_cfg()
  return replace(cfg, experiment_name=experiment_name, run_name=run_name)


register_mjlab_task(
  task_id="WujiHand2_ReorientPos_50Hz",
  env_cfg=_reorient_pos_hand2_50hz_env_cfg(_GOAL_POS_RANGE),
  play_env_cfg=_reorient_pos_hand2_50hz_env_cfg(_GOAL_POS_RANGE, play=True),
  rl_cfg=_reorient_pos_hand2_50hz_rl_cfg(
    "wuji_reorient_hand2_pos", "ReorientPos_Hand2_50Hz"
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="WujiHand2_ReorientPosFixed_50Hz",
  env_cfg=_reorient_pos_hand2_50hz_env_cfg(_GOAL_POS_FIXED),
  play_env_cfg=_reorient_pos_hand2_50hz_env_cfg(_GOAL_POS_FIXED, play=True),
  rl_cfg=_reorient_pos_hand2_50hz_rl_cfg(
    "wuji_reorient_hand2_pos_fixed", "ReorientPosFixed_Hand2_50Hz"
  ),
  runner_cls=WujiOnPolicyRunner,
)
