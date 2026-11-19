# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 grasp-stage env: one task whose object, initial grasps and goal
grasps come from a stage of a training file (see ``grasp_stage.schema``).

Built on the 50 Hz position-goal recipe, with the wrist fixed at the mount by
default (``wrist=True`` adds the actuated 6-DoF wrist). The goal
command draws a goal grasp per episode; the reset puts the hand and object
in an initial grasp. Observations add the masked tip errors, the grasp
weights, the joint goal and the object encoding; rewards add the weighted
tip and joint terms (no cage-pose penalty); the object speed limit guards against mujoco-warp's
free-body spin-up.
"""

from __future__ import annotations

from dataclasses import fields
from functools import partial
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
from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
  NEIGHBOR_TABLE,
  TrainingSpec,
  load_training,
)
from wuji_mjlab.tasks.reorient.mdp.graphed import graph_plain_terms, graphed_term

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
_JOINT_REWARD_WEIGHT = 10.0
# Warmup targets a little inside the initial grasp, so the hand grips the
# object instead of just touching it: thumb opposition and the flexion joints
# [rad]. On the cube_dex_3k grasps this cut the share of objects moving
# > 10 mm in the 0.4 s warmup from 75 % to 35 % (dropped: 2.1 % -> 0.5 %).
_WARMUP_SQUEEZE = {
  r".*_finger1_joint1": 0.2,
  r".*_finger1_joint[34]": 0.1,
  r".*_finger[2-5]_joint[134]": 0.1,
}
# Hand shape reward: exp(-e / scale) per finger. At the training start the
# finger joint errors are ~1 rad, where a 0.2 rad scale gives ~0 and no
# gradient; and the pose gate would hold it at 0 until the object is near its
# goal pose, so part of it (gate_floor) is always paid. Joint errors are
# scaled by the joint's importance (joints 1-4, base first: an error near the
# base moves everything above it, one near the tip changes the shape less)
# and each finger is judged by its largest one: with a mean over joints
# (joint_reduce="mean") the m2o policy left single joints ~0.9 rad off.
_JOINT_REWARD_PARAMS = {
  "scale": 0.6,
  "gate_floor": 0.3,
  "joint_importance": (1.0, 1.0, 0.5, 0.3),
  "joint_reduce": "max",
}


# Hand-object contacts: condim 6 (sliding, torsional and rolling friction)
# on the hand's collision geoms, which set the contact parameters (priority
# 1 over the object). Sliding friction drawn per env in [1.0, 2.0] (the
# middle without domain randomization); torsional 0.005 and rolling 0.0001
# stay at the hand defaults.
_HAND_CONDIM = 6
_HAND_SLIDING_FRICTION = (1.0, 2.0)


def _hand_contact_spec(spec_fn, condim: int, sliding: float):
  """``spec_fn()`` with every colliding geom at ``condim`` and sliding
  friction ``sliding``."""
  spec = spec_fn()
  for geom in spec.geoms:
    if geom.contype or geom.conaffinity:
      geom.condim = condim
      geom.friction[0] = sliding
  return spec


def build_stage_env_cfg(
  training: TrainingSpec,
  stage_index: int,
  play: bool = False,
  wrist: bool = False,
  glove: bool = False,
):
  """Env cfg for one stage. The wrist is fixed at the mount by default;
  ``wrist=True`` adds the actuated wrist (joints, actions, wrist curriculum).
  ``glove=True`` puts the glove covers on the hand (soft covers on hard
  links, hand2_model.apply_hand2_glove)."""
  stage = training.stages[stage_index]
  (obj,) = stage.objects  # one object per stage for now (validated)
  if obj.kind != "compound":
    raise NotImplementedError("grasp stage: mesh objects are not supported yet")
  cfg = _reorient_pos_hand2_50hz_env_cfg(((0.0, 0.0),) * 3, play=play, glove=glove)
  if wrist:
    cfg = _with_wrist(cfg)
  cfg.curriculum.pop("goal_rotation", None)  # goals come from the grasp set

  robot = cfg.scene.entities["robot"]
  robot.spec_fn = partial(
    _hand_contact_spec, robot.spec_fn, _HAND_CONDIM, sum(_HAND_SLIDING_FRICTION) / 2
  )
  if "robot_friction" in cfg.events:  # domain randomization (not in play)
    friction = cfg.events["robot_friction"].params
    friction.update(operation="abs", ranges=_HAND_SLIDING_FRICTION, axes=[0])

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
      "goal_neighbors": (
        str(stage.goal_set_folders[obj.name][0] / NEIGHBOR_TABLE)
        if stage.goal_sampling == "neighbors"
        else None
      ),
      "hand": training.hand,
      # A drop or a speed-limit end restarts on the same goal.
      "keep_goal_on_terminations": ("cage_drop", "object_speed_limit"),
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
  cfg.actions["finger_joint_pos"].warmup_squeeze = _WARMUP_SQUEEZE

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
        term.func = graphed_term(func)  # same parameters (injection) as before
        if corrupt and term.noise is not None:
          # Noise before the mask, so a part not required reads exactly 0.
          term.params = {**term.params, "noise": term.noise}
          term.noise = None
  history = cfg.observations["policy"].terms["cube_pos_in_tag"].history_length
  for group in ("policy", "critic"):
    terms = cfg.observations[group].terms
    noisy = cfg.observations[group].enable_corruption and group == "policy"
    terms["grasp_tip_error"] = ObservationTermCfg(
      func=graphed_term(gmdp.grasp_tip_error),
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
  # Pure tensor terms replay as CUDA graphs (mdp/graphed.py): one launch each
  # instead of tens of small kernels.
  cfg.rewards["orientation_alignment"].func = graphed_term(
    gmdp.grasp_orientation_alignment
  )
  cfg.rewards["position_alignment"].func = graphed_term(gmdp.grasp_position_alignment)
  cfg.rewards["grasp_tip"] = RewardTermCfg(
    func=graphed_term(gmdp.grasp_tip_proximity),
    weight=_TIP_REWARD_WEIGHT,
    params=command,
  )
  cfg.rewards["grasp_joint"] = RewardTermCfg(
    func=graphed_term(gmdp.grasp_joint_proximity),
    weight=_JOINT_REWARD_WEIGHT,
    params={**command, **_JOINT_REWARD_PARAMS},
  )
  # The reorient task's pull towards the cage pose works against goal hand
  # shapes; the hand shape is rewarded by grasp_joint alone.
  cfg.rewards.pop("hand_pose", None)
  cfg.terminations["object_speed_limit"] = TerminationTermCfg(
    func=graphed_term(gmdp.object_speed_limit)
  )
  # The speed limit ends the episode at once; charged like a drop (the
  # cage-escape penalty its remaining outside steps would cost), so flicking
  # the object is no way out of a failing episode.
  cage = cfg.rewards["cage_escape"]
  cfg.rewards["speed_limit_drop"] = RewardTermCfg(
    func=gmdp.speed_limit_drop_penalty,
    weight=cage.weight,
    params={
      "max_outside_steps": cage.params.get("max_outside_steps", 10),
      "drop_scale": cage.params.get("drop_scale", 4.0),
    },
  )
  # The other plain-function observation and metric terms as well (each
  # checked against eager with WUJI_GRAPH_CHECK=1). Eager: action_jerk_rms
  # swaps its previous delta in a Python attribute every step; dr_params
  # copies index lists to the GPU on every call; the contact-sensor metrics
  # read sensor data refreshed on the Python side.
  for group in ("policy", "critic"):
    graph_plain_terms(cfg.observations[group].terms, eager=("dr_params",))
  graph_plain_terms(
    cfg.metrics,
    eager=("action_jerk_rms", "finger_collision_frequency", "palm_detach_frequency"),
  )
  return cfg


def example_stage_env_cfg(play: bool = False):
  """Stage 0 of the bundled example training (for the task registry)."""
  return build_stage_env_cfg(load_training(EXAMPLE_TRAINING), 0, play=play)
