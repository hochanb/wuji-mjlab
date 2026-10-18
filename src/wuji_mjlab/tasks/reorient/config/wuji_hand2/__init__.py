# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 reorient task registration (right hand only)."""

from dataclasses import replace

from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  get_compound_object_cfg,
)
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient import mdp

from .env_cfgs import wuji_hand2_reorient_env_cfg
from .hand2_constants import REORIENT_HAND2_CUBE_INIT_POS
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


# Stapler (compound primitive object). Dropped from 4 cm above the cube's start in
# its upright pose: a 13 cm object at a uniformly random orientation would start
# interpenetrating the hand.
_STAPLER_INIT_LIFT = 0.04


def _with_stapler(cfg):
  """Swap the cube for the stapler in a reorient env cfg."""
  x, y, z = REORIENT_HAND2_CUBE_INIT_POS["right"]
  stapler = get_compound_object_cfg("stapler")
  stapler.init_state.pos = (x, y, z + _STAPLER_INIT_LIFT)
  stapler.init_state.rot = (1.0, 0.0, 0.0, 0.0)
  cfg.scene.entities["object"] = stapler
  # The cube sensors match the object geom named "cube"; on a multi-geom object
  # that is only the first part, so match the whole object body instead.
  for sensor in cfg.scene.sensors:
    if sensor.name == "tip_object_contact":
      sensor.secondary = ContactMatch(mode="body", pattern="cube", entity="object")
  # Size DR scales geom extents in place, which would pull the parts apart.
  cfg.events.pop("object_size", None)
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {k: (-0.01, 0.01) for k in ("x", "y", "z")},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("object"),
    },
  )
  return cfg


register_mjlab_task(
  task_id="WujiHand2_ReorientStapler_50Hz",
  env_cfg=_with_stapler(_reorient_hand2_50hz_env_cfg(num_envs=8192)),
  play_env_cfg=_with_stapler(_reorient_hand2_50hz_env_cfg(play=True)),
  rl_cfg=_reorient_pos_hand2_50hz_rl_cfg(
    "wuji_reorient_hand2_stapler", "ReorientStapler_Hand2_50Hz"
  ),
  runner_cls=WujiOnPolicyRunner,
)

# Stapler goal-position box: the cube box's size (_GOAL_POS_RANGE), re-centred on
# the median tag-frame position where the stapler settles on the idle hand.
_STAPLER_REST_POS = (-0.130, -0.006, 0.022)
_STAPLER_GOAL_POS_RANGE = tuple(
  (c - (hi - lo) / 2, c + (hi - lo) / 2)
  for c, (lo, hi) in zip(_STAPLER_REST_POS, _GOAL_POS_RANGE, strict=True)
)

register_mjlab_task(
  task_id="WujiHand2_ReorientStaplerPos_50Hz",
  env_cfg=_with_stapler(_reorient_pos_hand2_50hz_env_cfg(_STAPLER_GOAL_POS_RANGE)),
  play_env_cfg=_with_stapler(
    _reorient_pos_hand2_50hz_env_cfg(_STAPLER_GOAL_POS_RANGE, play=True)
  ),
  rl_cfg=_reorient_pos_hand2_50hz_rl_cfg(
    "wuji_reorient_hand2_stapler_pos", "ReorientStaplerPos_Hand2_50Hz"
  ),
  runner_cls=WujiOnPolicyRunner,
)
