# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Actuated 6-DoF wrist for Wuji Hand 2 (reorient).

A ``right_wrist_link`` body is inserted between the mocap mount (the entity
root, which keeps the reset mount tilt) and the palm. It carries three slide
joints and three hinge joints about the mount pivot, each driven by a stiff
position actuator, so the hand base moves with real velocities and contacts
see the motion (a kinematically moved mocap hand has zero velocity in the
contact solver and lets the object sink through).

Joint axes are the world axes of the untilted mount pose, expressed in the root
(palm) frame; with the reset tilt applied they tilt with the mount.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np

from .hand2_constants import REORIENT_HAND2_MOUNT_POS, REORIENT_HAND2_ROOT_ROT

WRIST_MAX_ANGLE = math.radians(15.0)  # [rad] per hinge
WRIST_MAX_TRANSLATION = 0.05  # [m] per slide
# Order matches the wrist action layout [rx, ry, rz, tx, ty, tz].
WRIST_JOINT_PATTERN = "right_wrist_[rt][xyz]"
WRIST_JOINT_NAMES = (
  "right_wrist_rx",
  "right_wrist_ry",
  "right_wrist_rz",
  "right_wrist_tx",
  "right_wrist_ty",
  "right_wrist_tz",
)
_WRIST_BODY = "right_wrist_link"
# Position servo: natural frequency / damping ratio on the joint's effective
# inertia. omega * dt = 0.8 at dt = 10 ms (damping is integrated implicitly).
_SERVO_OMEGA = 80.0  # [rad/s]
_SERVO_ZETA = 1.0
_SLIDE_ARMATURE = 0.2  # [kg]
_HINGE_ARMATURE = 0.002  # [kg m^2]


def _world_axes_in_root() -> np.ndarray:
  rot = np.empty(9)
  mujoco.mju_quat2Mat(rot, np.asarray(REORIENT_HAND2_ROOT_ROT, dtype=float))
  # R maps root -> world, so world axis i in the root frame is R^T e_i = row i of R.
  return rot.reshape(3, 3)


def add_wrist_joints(base_spec_fn: Callable[[], mujoco.MjSpec]) -> mujoco.MjSpec:
  """Wrap the hand spec in a 6-DoF actuated wrist link."""
  hand = base_spec_fn()
  for key in list(hand.keys):  # qpos size changes; init_state sets the pose
    hand.delete(key)

  spec = mujoco.MjSpec()
  spec.compiler.degree = False  # joint ranges below are in radians
  wrist = spec.worldbody.add_body(name=_WRIST_BODY)
  wrist.mass = 0.05
  wrist.inertia = [1e-5, 1e-5, 1e-5]
  wrist.explicitinertial = True
  axes = _world_axes_in_root()
  pivot = list(REORIENT_HAND2_MOUNT_POS)
  # Slides first (translate the whole wrist), then hinges about the pivot.
  for name, axis in zip(WRIST_JOINT_NAMES[3:], axes, strict=True):
    wrist.add_joint(
      name=name,
      type=mujoco.mjtJoint.mjJNT_SLIDE,
      axis=axis.tolist(),
      range=[-WRIST_MAX_TRANSLATION, WRIST_MAX_TRANSLATION],
      limited=mujoco.mjtLimited.mjLIMITED_TRUE,
      armature=_SLIDE_ARMATURE,
    )
  for name, axis in zip(WRIST_JOINT_NAMES[:3], axes, strict=True):
    wrist.add_joint(
      name=name,
      type=mujoco.mjtJoint.mjJNT_HINGE,
      axis=axis.tolist(),
      pos=pivot,
      range=[-WRIST_MAX_ANGLE, WRIST_MAX_ANGLE],
      limited=mujoco.mjtLimited.mjLIMITED_TRUE,
      armature=_HINGE_ARMATURE,
    )
  spec.attach(child=hand, prefix="", frame=wrist.add_frame())

  # Gains from each wrist dof's effective inertia at the default pose.
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  inertia = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, inertia)
  for name in WRIST_JOINT_NAMES:
    dof = model.jnt_dofadr[model.joint(name).id]
    m_eff = float(inertia[dof, dof])
    kp = _SERVO_OMEGA**2 * m_eff
    kv = 2.0 * _SERVO_ZETA * _SERVO_OMEGA * m_eff
    limit = WRIST_MAX_ANGLE if name in WRIST_JOINT_NAMES[:3] else WRIST_MAX_TRANSLATION
    act = spec.add_actuator(
      name=f"{name}_actuator",
      target=name,
      trntype=mujoco.mjtTrn.mjTRN_JOINT,
      ctrlrange=[-limit, limit],
      ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
    )
    act.set_to_position(kp=kp, kv=kv)
  return spec
