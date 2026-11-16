# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

import math
from collections.abc import Callable

import mujoco
from mjlab.entity import EntityCfg

from wuji_mjlab.assets.robots.wuji_hand2.hand2_geometry import (  # noqa: F401
  HAND2_TIP_PAD_DEPTH,
  HAND2_TIP_PAD_POINTS,
)
from wuji_mjlab.assets.robots.wuji_hand2.hand2_model import (  # noqa: F401
  HAND2_ACTUATOR_FORCE_LIMITS,
  HAND2_ACTUATOR_GAINS,
  HAND2_COLLISION_CAPSULES,
  HAND2_GEOM_DEFAULTS,
  HAND2_JIG_POS,
  HAND2_JIG_ROT,
  HAND2_MOUNT_POS,
  HAND2_MOUNT_ROT,
  HAND2_TAG_POS,
  HAND2_TAG_ROT,
  add_frame_axes,
  apply_hand2_actuator_gains,
  apply_hand2_collision_overlay,
  apply_hand2_contact,
  apply_hand2_force_limits,
  get_hand2_model_spec,
  inject_hand2_rig,
)

REORIENT_HAND2_PALM_NORMAL_AXIS = 1

# The hand model moved to wuji_mjlab.assets.robots.wuji_hand2.hand2_model; the
# reorient names stay as aliases.
HAND2_REORIENT_GEOM_DEFAULTS = HAND2_GEOM_DEFAULTS
HAND2_REORIENT_ACTUATOR_GAINS = HAND2_ACTUATOR_GAINS
HAND2_REORIENT_ACTUATOR_FORCE_LIMITS = HAND2_ACTUATOR_FORCE_LIMITS
HAND2_REORIENT_CAPSULES = HAND2_COLLISION_CAPSULES
apply_hand2_reorient_actuator_gains = apply_hand2_actuator_gains
apply_hand2_reorient_force_limits = apply_hand2_force_limits
apply_hand2_reorient_collision_overlay = apply_hand2_collision_overlay
inject_hand2_reorient_rig = inject_hand2_rig


REORIENT_HAND2_ROOT_POS = (-0.0284999, 0.00300004, 0.500250158)
REORIENT_HAND2_ROOT_ROT = (0.5, 0.5, 0.5, 0.5)

REORIENT_HAND2_MOUNT_POS = HAND2_MOUNT_POS
# Physical Hand 2 jig: hand pitched down 10 deg about world Y through the mount
# (the PenSpin RIGHT_FIXED_WRIST_POSE_W palm normal is at -10.00 deg in this
# convention). Used as the fixed reset tilt.
REORIENT_HAND2_MOUNT_PITCH = math.radians(-10.0)
REORIENT_HAND2_MOUNT_ROT = HAND2_MOUNT_ROT
REORIENT_HAND2_TAG_POS = HAND2_TAG_POS
REORIENT_HAND2_TAG_ROT = HAND2_TAG_ROT
REORIENT_HAND2_JIG_POS = HAND2_JIG_POS
REORIENT_HAND2_JIG_ROT = HAND2_JIG_ROT


def apply_hand2_reorient_physics(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """The hand's contact parameters and the reorient task's simulation
  options."""
  spec = apply_hand2_contact(spec)
  spec.option.timestep = 0.01
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  spec.option.iterations = 5
  spec.option.ls_iterations = 8
  spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
  spec.option.tolerance = 1e-8
  spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  return spec


def get_hand2_reorient_spec(
  base_spec_fn: Callable[[], mujoco.MjSpec],
) -> mujoco.MjSpec:
  """The hand on its calibration rig (``hand2_model``) with the reorient
  task's simulation options."""
  return apply_hand2_reorient_physics(get_hand2_model_spec(base_spec_fn))


# Cage joint pose (finger1=thumb ... finger5=pinky; joint2 = abduction).
REORIENT_HAND2_JOINT_POS: dict[str, float] = {
  ".*_finger1_joint1": 0.8,
  ".*_finger1_joint2": -0.3,
  ".*_finger1_joint3": 0.5,
  ".*_finger1_joint4": 0.4,
  ".*_finger2_joint1": 0.45,
  ".*_finger2_joint2": -0.15,
  ".*_finger2_joint3": 0.8,
  ".*_finger2_joint4": 0.5,
  ".*_finger3_joint1": 0.45,
  ".*_finger3_joint2": 0.0,
  ".*_finger3_joint3": 0.9,
  ".*_finger3_joint4": 0.3,
  ".*_finger4_joint1": 0.6,
  ".*_finger4_joint2": 0.15,
  ".*_finger4_joint3": 1.1,
  ".*_finger4_joint4": 0.2,
  ".*_finger5_joint1": 1.2,
  ".*_finger5_joint2": 0.2,
  ".*_finger5_joint3": 0.8,
  ".*_finger5_joint4": 0.4,
}

REORIENT_HAND2_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REORIENT_HAND2_ROOT_POS,
  rot=REORIENT_HAND2_ROOT_ROT,
  joint_pos=REORIENT_HAND2_JOINT_POS,
  joint_vel={".*": 0.0},
)

REORIENT_HAND2_CUBE_INIT_POS = {"right": (-0.108, 0.0072, 0.5616)}
REORIENT_HAND2_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def hand2_cube_init_state(hand_side: str) -> EntityCfg.InitialStateCfg:
  if hand_side != "right":
    raise ValueError(f"Unsupported hand_side '{hand_side}'. Only 'right' is supported.")
  return EntityCfg.InitialStateCfg(
    pos=REORIENT_HAND2_CUBE_INIT_POS[hand_side],
    rot=REORIENT_HAND2_CUBE_INIT_ROT,
  )


# Fingertip reference points: hand geometry, defined with the robot asset.
