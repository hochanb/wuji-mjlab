# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 reorient task registration (right hand only)."""

import math
from dataclasses import replace
from functools import partial

from mjlab.actuator import XmlActuatorCfg
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  get_compound_object_cfg,
  stapler_fingertip_targets,
)
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient import mdp

from .env_cfgs import wuji_hand2_reorient_env_cfg
from .hand2_constants import HAND2_TIP_PAD_POINTS, REORIENT_HAND2_CUBE_INIT_POS
from .rsl_rl.ppo import wuji_hand2_reorient_ppo_runner_cfg
from .wrist import (
  WRIST_JOINT_NAMES,
  WRIST_JOINT_PATTERN,
  WRIST_MAX_ANGLE,
  WRIST_MAX_TRANSLATION,
  add_wrist_joints,
)


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
  cfg.actions["finger_joint_pos"].ema_alpha = 0.3
  cfg.rewards["finger_action_rate"].weight = 0.0
  cfg.rewards["finger_action_hf"].weight = -30.0
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


# Policy std cap for the wrist tasks: actions are clipped to [-1, 1], so a larger
# std only adds clipped (bang-bang) samples rather than exploration.
_WRIST_TASK_STD_RANGE = (0.1, 1.0)


def _wrist_rl_cfg(experiment_name: str, run_name: str):
  cfg = _reorient_pos_hand2_50hz_rl_cfg(experiment_name, run_name)
  distribution = {**cfg.actor.distribution_cfg, "std_range": _WRIST_TASK_STD_RANGE}
  return replace(cfg, actor=replace(cfg.actor, distribution_cfg=distribution))


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


def _with_stapler(cfg, rot=(1.0, 0.0, 0.0, 0.0)):
  """Swap the cube for the stapler in a reorient env cfg (start pose ``rot``)."""
  x, y, z = REORIENT_HAND2_CUBE_INIT_POS["right"]
  stapler = get_compound_object_cfg("stapler")
  stapler.init_state.pos = (x, y, z + _STAPLER_INIT_LIFT)
  stapler.init_state.rot = rot
  cfg.scene.entities["object"] = stapler
  # Fingertip targets (shown in the viewer); the reward is off until weighted.
  cfg.commands["reorient_command"].fingertip_targets = stapler_fingertip_targets()
  cfg.commands["reorient_command"].tip_point_offsets = HAND2_TIP_PAD_POINTS
  cfg.rewards["fingertip_target"] = RewardTermCfg(
    func=mdp.fingertip_target_proximity,
    weight=0.0,
    params={
      "command_name": "reorient_command",
      "scale": 0.02,
      "pad_offset": 0.005,  # depth of HAND2_TIP_PAD_POINTS inside the pad
      "ori_bound": 0.2,
      "ori_margin": 0.3,
      "pos_bound": 0.01,
      "pos_margin": 0.02,
    },
  )
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


_FINGER_JOINTS = (".*_finger.*_joint.*",)


def _with_wrist(cfg):
  """Add an actuated 6-DoF wrist (+-15 deg about the mount pivot, +-5 cm per axis).

  The wrist joints sit between the mocap mount and the palm (see ``wrist.py``);
  terms that meant "all hand joints" are narrowed to the finger joints.
  """
  robot = cfg.scene.entities["robot"]
  robot.spec_fn = partial(add_wrist_joints, robot.spec_fn)
  robot.init_state = replace(
    robot.init_state,
    joint_pos={**robot.init_state.joint_pos, WRIST_JOINT_PATTERN: 0.0},
  )
  robot.articulation = replace(
    robot.articulation,
    actuators=(
      *robot.articulation.actuators,
      XmlActuatorCfg(target_names_expr=(WRIST_JOINT_PATTERN,)),
    ),
  )

  fingers = SceneEntityCfg("robot", joint_names=_FINGER_JOINTS)
  finger_actuators = SceneEntityCfg("robot", actuator_names=_FINGER_JOINTS)
  cfg.actions["finger_joint_pos"].actuator_names = _FINGER_JOINTS
  cfg.actions["wrist_pose"] = mdp.WristPoseActionCfg(
    entity_name="robot",
    joint_names=WRIST_JOINT_NAMES,
    max_angle=WRIST_MAX_ANGLE,
    max_translation=WRIST_MAX_TRANSLATION,
  )
  for group, names in (
    ("policy", ("noisy_joint_angles", "qpos_error")),
    ("critic", ("joint_angles", "qpos_error")),
  ):
    for name in names:
      cfg.observations[group].terms[name].params["asset_cfg"] = fingers
  cfg.rewards["hand_pose"].params["asset_cfg"] = fingers
  # Wrist regularisers. At 1/5 of these weights the wrist sat at its limits
  # (bang-bang) and shook; the regularisers were ~10% of the alignment reward.
  # Jerk of the EMA output before the rate limit: after it, the per-step change
  # is capped, so the penalty would saturate exactly when the wrist shakes most.
  cfg.rewards["wrist_action_hf"] = RewardTermCfg(
    func=mdp.ActionHighFreqPenalty,
    weight=-50.0,
    params={"action_name": "wrist_pose", "signal": "smoothed_action"},
  )
  cfg.rewards["wrist_pose"] = RewardTermCfg(
    func=mdp.wrist_pose_penalty, weight=-0.5, params={"action_name": "wrist_pose"}
  )
  # Keep the policy mean inside the clipped action range (see action_out_of_bounds).
  for name, key, weight in (
    ("finger_joint_pos", "finger_action_bound", -1.0),
    ("wrist_pose", "wrist_action_bound", -5.0),
  ):
    cfg.rewards[key] = RewardTermCfg(
      func=mdp.action_out_of_bounds, weight=weight, params={"action_name": name}
    )
  cfg.rewards["torque"].params["asset_cfg"] = finger_actuators
  cfg.metrics["torque_saturation_ratio"].params["asset_cfg"] = finger_actuators
  for name in ("joint_acceleration_rms", "joint_vel_rms"):
    cfg.metrics[name].params["asset_cfg"] = fingers

  history_length = cfg.observations["policy"].terms["cube_pos_in_tag"].history_length
  for group in ("policy", "critic"):
    cfg.observations[group].terms["wrist_pose"] = ObservationTermCfg(
      func=mdp.wrist_pose, history_length=history_length
    )
  # Fingers first: the wrist range and the goal rotation both start small and
  # grow with success (with a time floor). Play/eval clears curricula, so both
  # run at full range there. Curriculum entries are absent in play cfgs.
  if cfg.curriculum:
    cfg.curriculum["wrist_range"] = CurriculumTermCfg(
      func=mdp.wrist_range_curriculum, params={"action_name": "wrist_pose"}
    )
    cfg.curriculum["goal_rotation"] = CurriculumTermCfg(
      func=mdp.goal_rotation_curriculum, params={"command_name": "reorient_command"}
    )
  return cfg


register_mjlab_task(
  task_id="WujiHand2_ReorientWrist_50Hz",
  env_cfg=_with_wrist(_reorient_hand2_50hz_env_cfg(num_envs=8192)),
  play_env_cfg=_with_wrist(_reorient_hand2_50hz_env_cfg(play=True)),
  rl_cfg=_wrist_rl_cfg("wuji_reorient_hand2_wrist", "ReorientWrist_Hand2_50Hz"),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="WujiHand2_ReorientPosWrist_50Hz",
  env_cfg=_with_wrist(_reorient_pos_hand2_50hz_env_cfg(_GOAL_POS_RANGE)),
  play_env_cfg=_with_wrist(
    _reorient_pos_hand2_50hz_env_cfg(_GOAL_POS_RANGE, play=True)
  ),
  rl_cfg=_wrist_rl_cfg("wuji_reorient_hand2_pos_wrist", "ReorientPosWrist_Hand2_50Hz"),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="WujiHand2_ReorientStaplerPosWrist_50Hz",
  env_cfg=_with_wrist(
    _with_stapler(_reorient_pos_hand2_50hz_env_cfg(_STAPLER_GOAL_POS_RANGE))
  ),
  play_env_cfg=_with_wrist(
    _with_stapler(_reorient_pos_hand2_50hz_env_cfg(_STAPLER_GOAL_POS_RANGE, play=True))
  ),
  rl_cfg=_wrist_rl_cfg(
    "wuji_reorient_hand2_stapler_pos_wrist", "ReorientStaplerPosWrist_Hand2_50Hz"
  ),
  runner_cls=WujiOnPolicyRunner,
)


# Fixed stapler goal: lying on its side across the palm (long axis near the
# palm width), seated slightly diagonally. Built from the crosswise pose
# (centred laterally at the along-finger offset where the stapler settles
# lengthwise on the idle hand, base 2 mm above the palm collision surface) by
# moving -2 cm along the object's y axis, 0.6 cm down its z axis (the palm
# normal) and turning +10 deg about its z axis; then turned -90 deg about its
# long (x) axis onto its side and lowered 6.3 mm along the palm normal to keep
# that pose's palm clearance (-0.9 mm), and turned a further -15 deg about its
# y axis (now the inward palm normal, so the clearance is unchanged), then
# moved 1.5 cm along its z axis and 2 cm along its x axis (both in the palm
# plane), flipped 180 deg about its x axis onto its other side and turned a
# further +15 deg about its y axis (now the outward palm normal; palm gap
# 4.5 mm). Tag frame.
_STAPLER_CROSSWISE_POS = (-0.1305, 0.0093, -0.0052)
_STAPLER_CROSSWISE_QUAT = (0.2721, 0.2721, 0.6527, 0.6527)
# Start pose for the fixed-goal task: upright, turned 180 deg about its z axis
# (hinge end towards the other side), then by a random rotation of at most
# 45 deg. The COM is the body origin, so the drop point is unchanged; at 45 deg
# about 0.2% of the spawns touch the hand, by under 1 mm.
_STAPLER_FIXED_GOAL_INIT_ROT = (0.0, 0.0, 0.0, 1.0)
_STAPLER_FIXED_GOAL_INIT_MAX_ANGLE = math.radians(45.0)


def _with_random_start(cfg, max_angle):
  """Reset the object within ``max_angle`` of its start pose (+-1 cm), and keep
  the fingers at their reset pose through the action warmup instead of closing
  them to the cage pose while the object drops in."""
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=mdp.reset_object_orientation,
    mode="reset",
    params={"pos_noise": 0.01, "max_angle": max_angle},
  )
  cfg.actions["finger_joint_pos"].warmup_hold_reset_pose = True
  return cfg


def _with_fixed_goal(cfg, pos, quat):
  command = cfg.commands["reorient_command"]
  command.goal_pos_range = tuple((c, c) for c in pos)
  command.goal_quat_tag_fixed = quat
  return cfg


register_mjlab_task(
  task_id="WujiHand2_ReorientStaplerFixedGoalWrist_50Hz",
  env_cfg=_with_fixed_goal(
    _with_wrist(
      _with_random_start(
        _with_stapler(
          _reorient_pos_hand2_50hz_env_cfg(_STAPLER_GOAL_POS_RANGE),
          rot=_STAPLER_FIXED_GOAL_INIT_ROT,
        ),
        _STAPLER_FIXED_GOAL_INIT_MAX_ANGLE,
      )
    ),
    _STAPLER_CROSSWISE_POS,
    _STAPLER_CROSSWISE_QUAT,
  ),
  play_env_cfg=_with_fixed_goal(
    _with_wrist(
      _with_random_start(
        _with_stapler(
          _reorient_pos_hand2_50hz_env_cfg(_STAPLER_GOAL_POS_RANGE, play=True),
          rot=_STAPLER_FIXED_GOAL_INIT_ROT,
        ),
        _STAPLER_FIXED_GOAL_INIT_MAX_ANGLE,
      )
    ),
    _STAPLER_CROSSWISE_POS,
    _STAPLER_CROSSWISE_QUAT,
  ),
  rl_cfg=_wrist_rl_cfg(
    "wuji_reorient_hand2_stapler_fixed_goal_wrist",
    "ReorientStaplerFixedGoalWrist_Hand2_50Hz",
  ),
  runner_cls=WujiOnPolicyRunner,
)
