# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Grasp synthesis for a given object pose (after Dexonomy, arXiv:2504.18829).

Given an object mesh, its poses in the hand base (wrist = ``right_palm_link``)
frame, a grasp type and k, find k finger joint configurations per pose that
hold the object. Dexonomy fits the object to a grasp template and then
refines the hand in simulation; here the object pose is fixed, so the search
runs over the fingers. Every simulated hand is one mujoco-warp world, so all
attempts for all poses run in parallel.

Refinement (object pinned, no gravity): each hand contact of the grasp type
is pulled to its nearest object surface point by a virtual spring and its
surface normal is turned against the object normal, ``tau = J^T f +
J_rot^T m``; MuJoCo contacts (margin ``contact_margin``) keep the hand just
off the surface. One round per pose:

1. Explore: ``explore_worlds`` hands start at the initial joint angles plus
   ``explore_noise`` on the grasping fingers and are refined. The fingers are
   independent chains on a fixed palm, so every finger that ends touching the
   object (within ``touch_dist``) without penetrating it is a candidate.
2. Combine: random combinations of one candidate per grasping finger are
   scored by contact-force balance (``wrench.balance_contact_forces``: the
   net force and torque of the best forces within friction cones, at most
   ``max_wrench_residual``).
3. Verify: the best combinations are refined again together and must pass:
   no hand-object or hand-hand penetration deeper than ``max_penetration``,
   every grasping finger touching, joints within the soft limits, balance
   over the actual MuJoCo contacts;
   then a hold test with the env's physics (elliptic friction cones) and
   gravity in the ``hand_quat_w`` orientation:
   the position actuators hold the balanced forces (targets ``q + J^T f /
   kp``), the object is pinned for ``settle_steps`` and released for
   ``hold_steps`` and must stay within ``max_pos_drift`` / ``max_rot_drift``
   (with ``perturb_force`` > 0, also under that force along +-x, +-y, +-z).

An attempt is one verified combination. A pose gets at most
``k * attempts_per_sample`` attempts and returns its first k successes, fewer
if the budget runs out (empty when none).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import (
  _get_spec,
  wuji_hand2_xml_path,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  get_hand2_reorient_spec,
)

from .grasp_types import GRASP_TYPES, PAD_DEPTH, GraspType, resolve_contact
from .object_model import GraspObject, SurfaceQuery
from .wrench import balance_contact_forces

NUM_JOINTS = 20
PALM_BODY = "right_palm_link"
# Mildly open hand with the thumb turned towards the fingers.
DEFAULT_INIT_JOINT_POS: dict[str, float] = {
  ".*_finger1_joint1": 0.8,
  ".*_finger1_joint2": -0.3,
  ".*": 0.0,
}


@dataclass
class GraspSynthCfg:
  # Hand.
  init_joint_pos: Mapping[str, float] | Sequence[float] = field(
    default_factory=lambda: dict(DEFAULT_INIT_JOINT_POS)
  )
  """Initial finger joint angles: 20 values (finger1 joint1 .. finger5
  joint4) or {regex: value}, first match wins. Fingers not in the grasp type
  stay here."""
  hand_quat_w: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
  """Hand base (palm) orientation in the world, wxyz: the gravity direction
  (``gravity``). The default is the reorient env's hand, palm up."""
  soft_limit_factor: float = 0.9
  """Grasps keep the joints within this fraction of their range."""

  # Search.
  attempts_per_sample: int = 10
  """N: at most k * N verified attempts per pose."""
  explore_worlds: int = 512
  """Hands explored per pose and round."""
  explore_noise: float = 1.0
  """Uniform noise [rad] on the grasping fingers' initial joint angles."""
  start_redraws: int = 3
  """Times a grasping finger starting inside the object is re-drawn."""
  combinations: int = 2048
  """Random finger combinations scored per pose and round."""
  verify_oversample: int = 4
  """Combinations verified per round: this many per missing grasp."""
  num_worlds: int = 4096
  device: str = "cuda:0"
  seed: int = 0

  # Refinement.
  explore_steps: int = 300
  verify_steps: int = 150
  spring_stiffness: float = 100.0  # N/m
  max_spring_force: float = 0.5  # N per contact
  align_stiffness: float = 0.01  # N m (at 90 deg)
  retarget_every: int = 10
  joint_damping: float = 0.05  # N m s / rad
  contact_margin: float = 0.001
  """Refinement keeps the hand this far off the object (MuJoCo margin)."""
  touch_dist: float = 0.002
  """A finger touches the object within this distance."""
  contact_tol: float = 0.004
  """A grasp-type contact counts for the force balance when its surface is
  within this distance of the object."""

  # Filters.
  max_penetration: float = 0.001
  friction: float = 0.5
  min_force_share: float = 0.5
  """Share of the total normal force spread over the grasp-type contacts when
  scoring combinations, so every grasping finger pushes."""
  max_wrench_residual: float = 0.1
  max_contacts: int = 32
  """Hand-object contacts per hand used for the final balance."""

  # Validation.
  grip_force: float = 5.0
  """Total normal force [N] of the grip; the balanced contact forces carry
  the object's weight at this force."""
  gravity: float = 9.81
  """Gravity [m/s^2] in the hold test, along -z of the world (hand_quat_w):
  the grip must also hold the object's weight. 0 only tests that the grip
  alone keeps the object still (net force and torque zero). Two-point grasps
  rarely hold the weight: condim 3 contacts have no torsional friction."""
  settle_steps: int = 20
  hold_steps: int = 100
  max_pos_drift: float = 0.005
  max_rot_drift: float = math.radians(10.0)
  perturb_force: float = 0.0

  # Simulation (the reorient env's, except the friction cone).
  cone: str = "elliptic"
  impratio: float = 10.0
  """Elliptic cones with a high impratio keep frictional contacts from
  creeping (the env uses pyramidal cones, impratio 1)."""
  timestep: float = 0.01
  solver_iterations: int = 10
  ls_iterations: int = 20
  nconmax: int = 128
  njmax: int = 1000


@dataclass
class GraspSynthResult:
  """Per pose (length j): arrays with one row per grasp found (<= k)."""

  joint_names: tuple[str, ...]
  joint_pos: list[np.ndarray]
  """(n, 20) grasp configurations (hand just touching the object)."""
  joint_target: list[np.ndarray]
  """(n, 20) position targets that hold it (squeeze)."""
  wrench_residual: list[np.ndarray]
  attempts: np.ndarray
  """(j,) verified attempts per pose."""
  stats: dict[str, int] = field(default_factory=dict)
  """Counts summed over poses and rounds (diagnostics)."""


@dataclass
class _GraspCtx:
  """Grasp-type tensors: contacts (C) and the joints they move."""

  body: torch.Tensor  # (C,)
  point: torch.Tensor  # (C, 3) body frame
  normal: torch.Tensor  # (C, 3) body frame
  chain: torch.Tensor  # (C, 20) 1 where the joint moves the contact
  finger: torch.Tensor  # (C,) 1..5
  fingers: torch.Tensor  # (F,) grasping fingers
  active_joint: torch.Tensor  # (20,) joints of grasping fingers


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    (
      1 - 2 * (y * y + z * z),
      2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z - w * x),
      2 * (x * z - w * y),
      2 * (y * z + w * x),
      1 - 2 * (x * x + y * y),
    ),
    -1,
  ).reshape(*q.shape[:-1], 3, 3)


def _rot_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  dot = (a * b).sum(-1).abs().clamp(max=1.0)
  return 2.0 * torch.acos(dot)


class GraspSynthesizer:
  """Holds the hand config; ``generate`` runs one object, grasp type and k.

  The simulation is built for the object on first use and reused while the
  same object (and world count) is passed.
  """

  def __init__(self, cfg: GraspSynthCfg | None = None):
    self.cfg = cfg or GraspSynthCfg()
    self._sim_key = None

  # Model / simulation.

  def _build_model(self, obj: GraspObject) -> mujoco.MjModel:
    spec = get_hand2_reorient_spec(partial(_get_spec, wuji_hand2_xml_path("right")))
    palm = spec.body(PALM_BODY)
    palm.pos, palm.quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_free")
    body.mass = obj.mass
    body.ipos = obj.com.tolist()
    inertia = obj.inertia
    body.fullinertia = [
      inertia[0, 0], inertia[1, 1], inertia[2, 2],
      inertia[0, 1], inertia[0, 2], inertia[1, 2],
    ]  # fmt: skip
    body.explicitinertial = True
    for i, part in enumerate(obj.collision_parts):
      name = f"object_part{i}"
      mesh = spec.add_mesh(name=name)
      mesh.uservert = part.vertices.astype(np.float32).reshape(-1).tolist()
      mesh.userface = part.faces.astype(np.int32).reshape(-1).tolist()
      mesh.maxhullvert = obj.max_hull_vertices
      body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        density=0,
        condim=3,
        friction=[0.3, 0.005, 0.0001],
        solref=[0.015, 1.0],
        solimp=[0.85, 0.85, 0.001, 0.5, 2.0],
        group=3,
      )
    return spec.compile()

  def _setup(self, obj: GraspObject, nworld: int) -> None:
    key = (id(obj), nworld)
    if self._sim_key == key:
      return
    cfg, dev = self.cfg, self.cfg.device
    model = self._build_model(obj)
    self.model, self.obj, self.nworld = model, obj, nworld
    self.sim = Simulation(
      num_envs=nworld,
      cfg=SimulationCfg(
        nconmax=cfg.nconmax,
        njmax=cfg.njmax,
        mujoco=MujocoCfg(
          timestep=cfg.timestep,
          cone=cfg.cone,
          impratio=cfg.impratio,
          iterations=cfg.solver_iterations,
          ls_iterations=cfg.ls_iterations,
        ),
      ),
      model=model,
      device=dev,
    )
    self.surface = SurfaceQuery(obj, dev)
    self.obj_com = torch.tensor(obj.com, device=dev).float()

    self.joint_names = tuple(model.joint(j).name for j in range(NUM_JOINTS))
    if not all(
      re.fullmatch(r"right_finger[1-5]_joint[1-4]", n) for n in self.joint_names
    ):
      raise RuntimeError(f"unexpected hand joints {self.joint_names}")
    if not (model.actuator_trnid[:, 0] == np.arange(NUM_JOINTS)).all():
      raise RuntimeError("hand actuators must drive joints 0..19 in order")
    obj_joint = model.joint("object_free")
    self.obj_qpos = int(obj_joint.qposadr[0])
    self.obj_qvel = int(obj_joint.dofadr[0])
    self.obj_body = model.body("object").id
    self.joint_range = torch.tensor(model.jnt_range[:NUM_JOINTS], device=dev).float()
    mid = self.joint_range.mean(1)
    half = (self.joint_range[:, 1] - self.joint_range[:, 0]) / 2 * cfg.soft_limit_factor
    self.soft_range = torch.stack((mid - half, mid + half), 1)
    self.kp = torch.tensor(model.actuator_gainprm[:, 0], device=dev).float()
    # body_chain[b, j] = 1 where joint j moves body b.
    self.body_chain = torch.zeros(model.nbody, NUM_JOINTS, device=dev)
    for b in range(model.nbody):
      a = b
      while a > 0:
        for jid in np.flatnonzero(model.jnt_bodyid[:NUM_JOINTS] == a):
          self.body_chain[b, jid] = 1.0
        a = int(model.body_parentid[a])
    self.joint_finger = torch.tensor(
      [int(n[len("right_finger")]) for n in self.joint_names], device=dev
    )

    # Geoms: object parts, and the finger (0 = palm) of each hand geom.
    self.obj_geom = torch.zeros(model.ngeom, dtype=torch.bool, device=dev)
    self.geom_finger = torch.full((model.ngeom,), -1, dtype=torch.long, device=dev)
    for g in range(model.ngeom):
      body = model.body(int(model.geom_bodyid[g])).name
      if body == "object":
        self.obj_geom[g] = True
      elif m := re.match(r"right_finger(\d)", body):
        self.geom_finger[g] = int(m.group(1))
      elif body == PALM_BODY:
        self.geom_finger[g] = 0
    # Shared (not per-world) model fields, switched between the phases.
    self.margin = wp.to_torch(self.sim.wp_model.geom_margin)
    self.gap = wp.to_torch(self.sim.wp_model.geom_gap)
    self.gravity = wp.to_torch(self.sim.wp_model.opt.gravity)
    self.margin0, self.gap0 = self.margin.clone(), self.gap.clone()
    q = torch.tensor(cfg.hand_quat_w, dtype=torch.float32)
    g = _quat_to_mat(q / q.norm()).T @ torch.tensor([0.0, 0.0, -cfg.gravity])
    self.gravity_palm = g.to(dev)
    self._sim_key = key

  def _init_joint_pos(self) -> torch.Tensor:
    spec = self.cfg.init_joint_pos
    if isinstance(spec, Mapping):
      values = []
      for name in self.joint_names:
        match = [v for pat, v in spec.items() if re.fullmatch(pat, name)]
        if not match:
          raise ValueError(f"init_joint_pos: no value for {name}")
        values.append(match[0])
    else:
      values = list(spec)
      if len(values) != NUM_JOINTS:
        raise ValueError(f"init_joint_pos needs {NUM_JOINTS} values")
    q = torch.tensor(values, dtype=torch.float32, device=self.cfg.device)
    return q.clamp(self.joint_range[:, 0], self.joint_range[:, 1])

  def _grasp_ctx(self, gtype: GraspType) -> _GraspCtx:
    dev = self.cfg.device
    resolved = [resolve_contact(self.model, c) for c in gtype.contacts]
    body = torch.tensor([r.body_id for r in resolved], device=dev)
    fingers = torch.tensor(gtype.fingers, device=dev)
    return _GraspCtx(
      body=body,
      point=torch.tensor(np.array([r.point for r in resolved]), device=dev).float(),
      normal=torch.tensor(np.array([r.normal for r in resolved]), device=dev).float(),
      chain=self.body_chain[body],
      finger=torch.tensor([r.finger for r in resolved], device=dev),
      fingers=fingers,
      active_joint=torch.isin(self.joint_finger, fingers),
    )

  # Kinematics on the current sim state.

  def _contact_frames(self, ctx: _GraspCtx) -> tuple[torch.Tensor, torch.Tensor]:
    """Grasp-type contact points and outward normals, world, (B, C, 3) each."""
    data = self.sim.data
    rot = data.xmat[:, ctx.body].reshape(self.nworld, -1, 3, 3)
    p = data.xpos[:, ctx.body] + torch.einsum("bcij,cj->bci", rot, ctx.point)
    return p, torch.einsum("bcij,cj->bci", rot, ctx.normal)

  def _joint_torque(self, points, forces, chain) -> torch.Tensor:
    """J^T f for forces at hand points, (B, C, 3) x2 -> (B, 20); ``chain``
    (C, 20) or (B, C, 20)."""
    axis = self.sim.data.xaxis[:, :NUM_JOINTS]
    anchor = self.sim.data.xanchor[:, :NUM_JOINTS]
    lever = points[:, :, None] - anchor[:, None]
    moment = torch.cross(lever, forces[:, :, None].expand_as(lever), dim=-1)
    return ((moment * axis[:, None]).sum(-1) * chain).sum(1)

  def _joint_moment(self, moments, chain) -> torch.Tensor:
    """J_rot^T m for moments on the contact links, (B, C, 3) -> (B, 20)."""
    axis = self.sim.data.xaxis[:, :NUM_JOINTS]
    return (torch.einsum("bjk,bck->bcj", axis, moments) * chain).sum(1)

  def _object_surface(self, points, pose):
    """Nearest object surface points to world ``points`` (B, C, 3): point,
    outward normal (world) and signed distance."""
    pos, rot = pose[:, :3], _quat_to_mat(pose[:, 3:])
    local = torch.einsum("bji,bcj->bci", rot, points - pos[:, None])
    near, normal, dist = self.surface.closest(local)
    near = torch.einsum("bij,bcj->bci", rot, near) + pos[:, None]
    return near, torch.einsum("bij,bcj->bci", rot, normal), dist

  def _pin_object(self, pose: torch.Tensor) -> None:
    data = self.sim.data
    data.qpos[:, self.obj_qpos : self.obj_qpos + 7] = pose
    data.qvel[:, self.obj_qvel : self.obj_qvel + 6] = 0.0

  # Simulation phases on one chunk of nworld hands.

  def _refine(self, pose, q0, ctx: _GraspCtx, steps: int) -> dict[str, torch.Tensor]:
    """Spring the grasp-type contacts onto the pinned object, then report
    contacts out to ``touch_dist``."""
    cfg, sim, data = self.cfg, self.sim, self.sim.data
    nw, dev = self.nworld, cfg.device
    self.gravity[:] = 0.0
    self.margin[:] = cfg.contact_margin
    self.gap[:] = 0.0
    data.qpos[:, :NUM_JOINTS] = q0
    data.qvel[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0
    self._pin_object(pose)
    sim.forward()
    active = ctx.active_joint
    for step in range(steps):
      self._pin_object(pose)
      data.ctrl[:] = torch.where(active, data.qpos[:, :NUM_JOINTS], q0)
      p_hand, n_hand = self._contact_frames(ctx)
      if step % cfg.retarget_every == 0:
        near, normal, _ = self._object_surface(p_hand, pose)
        # The reference point sits PAD_DEPTH off the surface when touching.
        target = near + PAD_DEPTH * normal
      spring = cfg.spring_stiffness * (target - p_hand)
      norm = spring.norm(dim=-1, keepdim=True).clamp_min(1e-9)
      spring = spring * (cfg.max_spring_force / norm).clamp(max=1.0)
      # Turn each contact's surface normal against the object normal.
      align = cfg.align_stiffness * torch.cross(n_hand, -normal, dim=-1)
      tau = self._joint_torque(p_hand, spring, ctx.chain)
      tau = tau + self._joint_moment(align, ctx.chain)
      tau = tau - cfg.joint_damping * data.qvel[:, :NUM_JOINTS]
      data.qfrc_applied[:, :NUM_JOINTS] = torch.where(active, tau, 0.0)
      sim.step()
    data.qfrc_applied[:] = 0.0

    # Contacts out to touch_dist, without contact forces (gap = margin).
    self._pin_object(pose)
    data.qvel[:] = 0.0
    self.margin[:] = cfg.touch_dist
    self.gap[:] = cfg.touch_dist
    sim.forward()
    q = data.qpos[:, :NUM_JOINTS].clone()
    n = int(data.nacon[0])
    world = data.contact.worldid[:n].long()
    geom = data.contact.geom[:n].long()
    dist = data.contact.dist[:n]
    obj1, obj2 = self.obj_geom[geom[:, 0]], self.obj_geom[geom[:, 1]]
    hand_obj = obj1 ^ obj2
    hand_geom = torch.where(obj1, geom[:, 1], geom[:, 0])
    finger = torch.where(hand_obj, self.geom_finger[hand_geom], 6)  # 6: not hand-object
    finger_depth = self._max_per(world, finger, -dist, nw, 7)
    touch = self._max_per(world, finger, (dist < cfg.touch_dist).float(), nw, 7) > 0
    hand_hand = (self.geom_finger[geom[:, 0]] >= 0) & (
      self.geom_finger[geom[:, 1]] >= 0
    )
    self_depth = self._max_per(
      world[hand_hand], torch.zeros_like(world[hand_hand]), -dist[hand_hand], nw, 1
    )[:, 0]

    p_hand, _ = self._contact_frames(ctx)
    near, normal, sdist = self._object_surface(p_hand, pose)
    within = (q >= self.soft_range[:, 0]) & (q <= self.soft_range[:, 1])
    finger_within = torch.ones(nw, 6, dtype=torch.bool, device=dev)
    for f in range(1, 6):
      finger_within[:, f] = within[:, self.joint_finger == f].all(1)
    return dict(
      q=q,
      pen=finger_depth[:, :6],  # deepest penetration per finger (0 = palm) [m]
      self_pen=self_depth,
      touch=touch[:, :6],
      within=finger_within,
      near=near,
      inward=-normal,
      in_contact=(sdist - PAD_DEPTH) < cfg.contact_tol,
    )

  def _object_contacts(self, max_dist: float, slots: int):
    """Current hand-object contacts within ``max_dist``, padded per world:
    points, inward normals (into the object), chains (B, S, 20), mask."""
    data, nw, dev = self.sim.data, self.nworld, self.cfg.device
    n = int(data.nacon[0])
    world = data.contact.worldid[:n].long()
    geom = data.contact.geom[:n].long()
    obj1, obj2 = self.obj_geom[geom[:, 0]], self.obj_geom[geom[:, 1]]
    keep = (obj1 ^ obj2) & (data.contact.dist[:n] < max_dist)
    world, geom, obj1 = world[keep], geom[keep], obj1[keep]
    pos = data.contact.pos[:n][keep]
    normal = data.contact.frame[:n][keep][:, 0]  # from geom[0] to geom[1]
    inward = torch.where(obj1[:, None], -normal, normal)
    body = torch.tensor(self.model.geom_bodyid, device=dev)[
      torch.where(obj1, geom[:, 1], geom[:, 0])
    ]
    order = torch.argsort(world, stable=True)
    world, pos, inward, body = world[order], pos[order], inward[order], body[order]
    start = torch.zeros(nw + 1, dtype=torch.long, device=dev)
    start[1:] = torch.bincount(world, minlength=nw).cumsum(0)
    slot = torch.arange(len(world), device=dev) - start[world]
    keep = slot < slots
    world, slot = world[keep], slot[keep]
    points = torch.zeros(nw, slots, 3, device=dev)
    normals = torch.zeros(nw, slots, 3, device=dev)
    chains = torch.zeros(nw, slots, NUM_JOINTS, device=dev)
    mask = torch.zeros(nw, slots, dtype=torch.bool, device=dev)
    points[world, slot] = pos[keep]
    normals[world, slot] = inward[keep]
    chains[world, slot] = self.body_chain[body[keep]]
    mask[world, slot] = True
    return points, normals, chains, mask

  @staticmethod
  def _max_per(world, group, value, nw, ngroup) -> torch.Tensor:
    out = torch.zeros(nw * ngroup, device=value.device)
    return out.scatter_reduce(0, world * ngroup + group, value, "amax").reshape(
      nw, ngroup
    )

  def _hold(self, pose, q_grasp, q_target) -> torch.Tensor:
    """Hold test with the env's contacts, gravity and actuators."""
    cfg, sim, data, dev = self.cfg, self.sim, self.sim.data, self.cfg.device
    data.qfrc_applied[:] = 0.0
    pushes = [torch.zeros(3, device=dev)]
    if cfg.perturb_force > 0:
      for axis in range(3):
        for sign in (1.0, -1.0):
          push = torch.zeros(3, device=dev)
          push[axis] = sign * cfg.perturb_force
          pushes.append(push)
    hold = torch.ones(self.nworld, dtype=torch.bool, device=dev)
    for push in pushes:
      data.qpos[:, :NUM_JOINTS] = q_grasp
      data.qvel[:] = 0.0
      data.ctrl[:] = q_target
      data.xfrc_applied[:] = 0.0
      for _ in range(cfg.settle_steps):
        self._pin_object(pose)
        sim.step()
      data.xfrc_applied[:, self.obj_body, :3] = push
      for _ in range(cfg.hold_steps):
        sim.step()
      obj_q = data.qpos[:, self.obj_qpos : self.obj_qpos + 7]
      drift = (obj_q[:, :3] - pose[:, :3]).norm(dim=-1)
      turn = _rot_angle(obj_q[:, 3:], pose[:, 3:])
      ok = (drift <= cfg.max_pos_drift) & (turn <= cfg.max_rot_drift)
      hold &= ok & torch.isfinite(obj_q).all(1)
    data.xfrc_applied[:] = 0.0
    return hold

  def _balance(self, near, inward, in_contact, pose, min_share):
    """Contact forces (total normal 1) that also carry the object's weight
    at ``grip_force`` total normal force."""
    center = pose[:, :3] + torch.einsum(
      "bij,j->bi", _quat_to_mat(pose[:, 3:]), self.obj_com
    )
    target = torch.zeros(len(pose), 6, device=pose.device)
    target[:, :3] = -self.obj.mass * self.gravity_palm / self.cfg.grip_force
    return balance_contact_forces(
      near, inward, in_contact, center, self.obj.radius, self.cfg.friction,
      min_share=min_share, target=target,
    )  # fmt: skip

  def _chunked(self, fn, pose, q0, *args):
    """Run ``fn(pose, q0, *args) -> dict`` over chunks of nworld rows."""
    n, out = len(pose), []
    for s in range(0, n, self.nworld):
      e = min(n, s + self.nworld)
      pad = self.nworld - (e - s)
      p = torch.cat((pose[s:e], pose[s : s + 1].expand(pad, -1)))
      q = torch.cat((q0[s:e], q0[s : s + 1].expand(pad, -1)))
      out.append({k: v[: e - s] for k, v in fn(p, q, *args).items()})
    return {k: torch.cat([o[k] for o in out]) for k in out[0]}

  def _verify(self, pose, q0, ctx: _GraspCtx) -> dict[str, torch.Tensor]:
    cfg = self.cfg
    r = self._refine(pose, q0, ctx, cfg.verify_steps)
    fingers = ctx.fingers
    no_pen = (r["pen"].amax(1) <= cfg.max_penetration) & (
      r["self_pen"] <= cfg.max_penetration
    )
    touch = r["touch"][:, fingers].all(1)
    limits = r["within"][:, fingers].all(1)
    # Balance over the actual contacts (any hand link, palm included).
    points, inward, chains, mask = self._object_contacts(
      cfg.touch_dist, cfg.max_contacts
    )
    # Incidental contacts (finger sides, palm) need not carry force here.
    residual, forces = self._balance(points, inward, mask, pose, 0.0)
    balanced = residual <= cfg.max_wrench_residual
    # Squeeze targets: the actuators supply J^T f and the hand's weight at
    # the grasp pose (env contacts and gravity from here on).
    tau = self._joint_torque(points, cfg.grip_force * forces, chains)
    q_grasp = r["q"]
    self.margin.copy_(self.margin0)
    self.gap.copy_(self.gap0)
    self.gravity[:] = self.gravity_palm
    self.sim.data.qvel[:] = 0.0
    self.sim.forward()
    tau = tau + self.sim.data.qfrc_bias[:, :NUM_JOINTS]
    q_target = torch.where(ctx.active_joint, q_grasp + tau / self.kp, q_grasp)
    q_target = q_target.clamp(self.joint_range[:, 0], self.joint_range[:, 1])
    passed = no_pen & touch & limits & balanced
    hold = self._hold(pose, q_grasp, q_target)
    return dict(
      q=q_grasp,
      q_target=q_target,
      residual=residual,
      no_pen=no_pen,
      touch=no_pen & touch,
      limits=no_pen & touch & limits,
      balanced=passed,
      ok=passed & hold,
    )

  def _explore(self, pose, q0, ctx: _GraspCtx, q_init, gen) -> dict[str, torch.Tensor]:
    # Re-draw the start of fingers that begin inside the object.
    cfg, data = self.cfg, self.sim.data
    self.margin.copy_(self.margin0)
    self.gap.copy_(self.gap0)
    for _ in range(cfg.start_redraws):
      data.qpos[:, :NUM_JOINTS] = q0
      self._pin_object(pose)
      self.sim.forward()
      n = int(data.nacon[0])
      world, geom = data.contact.worldid[:n].long(), data.contact.geom[:n].long()
      obj1 = self.obj_geom[geom[:, 0]]
      hand_obj = obj1 ^ self.obj_geom[geom[:, 1]]
      finger = self.geom_finger[torch.where(obj1, geom[:, 1], geom[:, 0])]
      deep = hand_obj & (data.contact.dist[:n] < 0)
      inside = torch.zeros(self.nworld, 6, dtype=torch.bool, device=cfg.device)
      inside[world[deep], finger[deep]] = True
      redraw = inside[:, self.joint_finger] & ctx.active_joint
      if not redraw.any():
        break
      fresh = self._explore_start(len(q0), ctx, q_init, gen)
      q0 = torch.where(redraw, fresh, q0)
    r = self._refine(pose, q0, ctx, cfg.explore_steps)
    r["ok"] = r["touch"] & r["within"] & (r["pen"] <= self.cfg.max_penetration)
    return r

  def _explore_start(self, n, ctx: _GraspCtx, q_init, gen) -> torch.Tensor:
    noise = torch.rand(n, NUM_JOINTS, device=self.cfg.device, generator=gen) * 2 - 1
    q0 = torch.where(ctx.active_joint, q_init + self.cfg.explore_noise * noise, q_init)
    return q0.clamp(self.soft_range[:, 0], self.soft_range[:, 1])

  def _combine(self, cand: dict, pose: torch.Tensor, ctx: _GraspCtx, q_init, gen):
    """Score random finger combinations from explored hands of one pose.

    Returns joint configurations (M', 20) of the balanced combinations, best
    first, and their residuals."""
    cfg, dev = self.cfg, self.cfg.device
    m = cfg.combinations
    pick = torch.empty(m, 6, dtype=torch.long, device=dev)
    for f in ctx.fingers.tolist():
      ok = cand["ok"][:, f].float()
      if ok.sum() == 0:
        return torch.empty(0, NUM_JOINTS, device=dev), torch.empty(0, device=dev)
      pick[:, f] = torch.multinomial(ok, m, replacement=True, generator=gen)
    src = pick[:, ctx.finger]  # (M, C) explored hand of each contact
    c = torch.arange(len(ctx.finger), device=dev)
    near, inward = cand["near"][src, c], cand["inward"][src, c]
    in_contact = cand["in_contact"][src, c]
    residual, _ = self._balance(
      near, inward, in_contact, pose.expand(m, -1), cfg.min_force_share
    )
    q = q_init.expand(m, -1).clone()
    for f in ctx.fingers.tolist():
      cols = self.joint_finger == f
      q[:, cols] = cand["q"][pick[:, f]][:, cols]
    keep = residual <= cfg.max_wrench_residual
    q, residual = q[keep], residual[keep]
    q, inverse = torch.unique(q, dim=0, return_inverse=True)
    best = torch.full((len(q),), float("inf"), device=dev)
    best = best.scatter_reduce(0, inverse, residual, "amin")
    order = torch.argsort(best)
    return q[order], best[order]

  # Main entry.

  def generate(
    self,
    obj: GraspObject,
    poses: np.ndarray | Sequence[Sequence[float]],
    grasp_type: str | GraspType,
    k: int,
  ) -> GraspSynthResult:
    """``poses`` (j, 7): object (mesh frame) pos + quat wxyz in the palm frame."""
    cfg, dev = self.cfg, self.cfg.device
    gtype = GRASP_TYPES[grasp_type] if isinstance(grasp_type, str) else grasp_type
    poses_np = np.asarray(poses, dtype=np.float64).reshape(-1, 7).copy()
    poses_np[:, 3:] /= np.linalg.norm(poses_np[:, 3:], axis=1, keepdims=True)
    j = len(poses_np)
    budget = k * cfg.attempts_per_sample
    self._setup(obj, min(cfg.num_worlds, j * max(cfg.explore_worlds, budget)))
    ctx = self._grasp_ctx(gtype)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    pose_all = torch.tensor(poses_np, device=dev).float()
    q_init = self._init_joint_pos()

    found: list[list[tuple]] = [[] for _ in range(j)]
    attempts = np.zeros(j, dtype=np.int64)
    stats = dict.fromkeys(
      ("explored", "candidates", "balanced", "verified", "no_pen", "touch",
       "limits", "wrench", "hold"), 0,
    )  # fmt: skip
    while pending := [
      p for p in range(j) if len(found[p]) < k and attempts[p] < budget
    ]:
      # 1. Explore.
      e = cfg.explore_worlds
      pidx = torch.tensor(pending, device=dev).repeat_interleave(e)
      q0 = self._explore_start(len(pidx), ctx, q_init, gen)
      cand = self._chunked(self._explore, pose_all[pidx], q0, ctx, q_init, gen)
      stats["explored"] += len(pidx)
      stats["candidates"] += int(cand["ok"][:, ctx.fingers].all(1).sum())

      # 2. Combine, per pose.
      v_pose, v_q = [], []
      for i, p in enumerate(pending):
        rows = slice(i * e, (i + 1) * e)
        sub = {key: val[rows] for key, val in cand.items()}
        q, _ = self._combine(sub, pose_all[p], ctx, q_init, gen)
        need = k - len(found[p])
        n = min(len(q), budget - attempts[p], need * cfg.verify_oversample)
        stats["balanced"] += len(q)
        # A round without a balanced combination spends the attempts it would
        # have verified.
        spent = n if n > 0 else need * cfg.verify_oversample
        attempts[p] += min(spent, budget - attempts[p])
        v_pose += [p] * n
        v_q.append(q[:n])
      if not v_pose:
        continue

      # 3. Verify.
      vidx = torch.tensor(v_pose, device=dev)
      r = self._chunked(self._verify, pose_all[vidx], torch.cat(v_q), ctx)
      stats["verified"] += len(vidx)
      for s in ("no_pen", "touch", "limits"):
        stats[s] += int(r[s].sum())
      stats["wrench"] += int(r["balanced"].sum())
      stats["hold"] += int(r["ok"].sum())
      ok = r["ok"].cpu().numpy()
      q, qt, res = (r[s].cpu().numpy() for s in ("q", "q_target", "residual"))
      for w in np.flatnonzero(ok):
        p = v_pose[w]
        if len(found[p]) < k:
          found[p].append((q[w], qt[w], res[w]))

    def stack(i, width):
      return [
        np.array([g[i] for g in f], dtype=np.float32).reshape(-1, *width) for f in found
      ]

    return GraspSynthResult(
      joint_names=self.joint_names,
      joint_pos=stack(0, (NUM_JOINTS,)),
      joint_target=stack(1, (NUM_JOINTS,)),
      wrench_residual=stack(2, ()),
      attempts=attempts,
      stats=stats,
    )
