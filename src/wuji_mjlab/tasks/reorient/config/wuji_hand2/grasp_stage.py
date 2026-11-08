# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 grasp-stage env: one task whose object, initial grasps and goal
grasps come from a stage of a training file (see ``grasp_stage.schema``).

Built on the 50 Hz position-goal recipe, with the wrist fixed at the mount by
default (``wrist=True`` adds the actuated 6-DoF wrist). The goal
command draws a goal grasp per episode; the reset puts the hand and object
in an initial grasp. Observations add the masked tip errors, the grasp
weights, the joint goal and the object encoding; rewards add the weighted
tip and joint terms; the object speed limit guards against mujoco-warp's
free-body spin-up.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  get_compound_object_cfg,
)
from wuji_mjlab.tasks.reorient.grasp_stage import mdp as gmdp
from wuji_mjlab.tasks.reorient.grasp_stage.command import GraspGoalCommandCfg
from wuji_mjlab.tasks.reorient.grasp_stage.schema import TrainingSpec, load_training

from . import _reorient_pos_hand2_50hz_env_cfg, _with_wrist
from .hand2_constants import HAND2_TIP_PAD_POINTS, REORIENT_HAND2_CUBE_INIT_POS

EXAMPLE_TRAINING = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "grasp_stage"
  / "trainings"
  / "stapler_single.yaml"
)
# Policy tip-error noise [m]: the object pose estimate and the joint readings
# both feed it; same as the position-error noise.
_TIP_ERROR_NOISE = Unoise(n_min=-0.008, n_max=0.008)
_TIP_REWARD_WEIGHT = 5.0
_JOINT_REWARD_WEIGHT = 5.0


def build_stage_env_cfg(
  training: TrainingSpec, stage_index: int, play: bool = False, wrist: bool = False
):
  """Env cfg for one stage. The wrist is fixed at the mount by default;
  ``wrist=True`` adds the actuated wrist (joints, actions, wrist curriculum)."""
  stage = training.stages[stage_index]
  (obj,) = stage.objects  # one object per stage for now (validated)
  if obj.kind != "compound":
    raise NotImplementedError("grasp stage: mesh objects are not supported yet")
  cfg = _reorient_pos_hand2_50hz_env_cfg(((0.0, 0.0),) * 3, play=play)
  if wrist:
    cfg = _with_wrist(cfg)
  cfg.curriculum.pop("goal_rotation", None)  # goals come from the grasp set

  # Object.
  x, y, z = REORIENT_HAND2_CUBE_INIT_POS["right"]
  entity = get_compound_object_cfg(obj.compound)
  entity.init_state.pos = (x, y, z + 0.04)  # overwritten by the grasp reset
  cfg.scene.entities["object"] = entity
  for sensor in cfg.scene.sensors:
    if sensor.name == "tip_object_contact":
      sensor.secondary = ContactMatch(mode="body", pattern="cube", entity="object")
  cfg.events.pop("object_size", None)  # scaling extents would pull parts apart

  # Goal command: a goal grasp per episode. Grasp sets are passed by folder;
  # the command and the reset read the arrays straight onto the device.
  goal_sets = tuple(str(p) for p in stage.goal_set_folders[obj.name])
  init_sets = tuple(str(p) for p in stage.init_set_folders[obj.name])
  base = cfg.commands["reorient_command"]
  cfg.commands["reorient_command"] = GraspGoalCommandCfg(
    **{f.name: getattr(base, f.name) for f in fields(base)}
    | {
      "goal_quat_tag_fixed": None,
      "tip_point_offsets": HAND2_TIP_PAD_POINTS,
      "pad_depth": training.hand_spec.pad_depth,
      "goal_grasp_sets": goal_sets,
      "hand": training.hand,
    }
  )

  # Reset into an initial grasp (fingers and object); the fingers hold it
  # through the action warmup.
  cfg.events.pop("reset_robot_joints", None)
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=gmdp.reset_to_init_grasp,
    mode="reset",
    params={"grasp_sets": init_sets, "hand": training.hand},
  )
  cfg.actions["finger_joint_pos"].warmup_hold_reset_pose = True

  # Observations: identical structure in every stage of a training. Pose
  # errors are zeroed for goals that do not require that part.
  command = {"command_name": "reorient_command"}
  pose_obs = {
    "cube_ori_error": gmdp.grasp_rot_err_6d,
    "cube_ori_error_clean": gmdp.grasp_rot_err_6d,
    "goal_pos_err": gmdp.grasp_pos_err,
    "goal_pos_err_clean": gmdp.grasp_pos_err,
  }
  for group in ("policy", "critic"):
    corrupt = cfg.observations[group].enable_corruption
    for name, func in pose_obs.items():
      term = cfg.observations[group].terms.get(name)
      if term is not None:
        term.func = func  # same parameters (injection) as before
        if corrupt and term.noise is not None:
          # Noise before the mask, so a part not required reads exactly 0.
          term.params = {**term.params, "noise": term.noise}
          term.noise = None
  history = cfg.observations["policy"].terms["cube_pos_in_tag"].history_length
  for group in ("policy", "critic"):
    terms = cfg.observations[group].terms
    noisy = cfg.observations[group].enable_corruption and group == "policy"
    terms["grasp_tip_error"] = ObservationTermCfg(
      func=gmdp.grasp_tip_error,
      params={**command, "noise": _TIP_ERROR_NOISE} if noisy else command,
      history_length=history,
    )
    terms["grasp_weights"] = ObservationTermCfg(func=gmdp.grasp_weights, params=command)
    terms["grasp_joint_goal"] = ObservationTermCfg(
      func=gmdp.grasp_joint_goal, params=command
    )
    if training.encoding_dim > 0:
      terms["object_encoding"] = ObservationTermCfg(
        func=gmdp.object_encoding, params={"encoding": obj.encoding}
      )
    for name in gmdp.CONDITION_TERMS:  # [env obs, condition]
      if name in terms:
        terms[name] = terms.pop(name)
      if name in terms and terms[name].noise is not None:
        raise ValueError(f"grasp stage: condition term {name} must not be noised")

  # Pose rewards scale with w_pos / w_rot. Both grasp schemes are always
  # present; the weights switch them on.
  cfg.rewards["orientation_alignment"].func = gmdp.grasp_orientation_alignment
  cfg.rewards["position_alignment"].func = gmdp.grasp_position_alignment
  cfg.rewards["grasp_tip"] = RewardTermCfg(
    func=gmdp.grasp_tip_proximity, weight=_TIP_REWARD_WEIGHT, params=command
  )
  cfg.rewards["grasp_joint"] = RewardTermCfg(
    func=gmdp.grasp_joint_proximity, weight=_JOINT_REWARD_WEIGHT, params=command
  )
  cfg.terminations["object_speed_limit"] = TerminationTermCfg(
    func=gmdp.object_speed_limit
  )
  return cfg


def example_stage_env_cfg(play: bool = False):
  """Stage 0 of the bundled example training (for the task registry)."""
  return build_stage_env_cfg(load_training(EXAMPLE_TRAINING), 0, play=play)
