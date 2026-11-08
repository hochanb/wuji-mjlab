# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Dexonomy grasp synthesis (arXiv:2504.18829): object poses from a hand shape.

Given an object mesh, a grasp type, a hand shape (finger joint angles, the
template) and k, find k grasps: the object pose in the wrist frame and the
finger joint angles that hold it. The object stays at the identity; the
search places the hand around it.

1. Global alignment (object frame, batched on the GPU). Each sample starts by
   putting one template contact on a random object surface point, normals
   opposed, with a random turn about the normal; the hand pose is then
   optimised (Adam) so that every contact lies on the surface with its
   normal against the object normal:
   ``E = w_p sum |p_i - (o_i + d n_i)|^2 + w_n sum |n_hand_i + n_i|^2`` with
   ``o_i`` / ``n_i`` the nearest surface point / outward normal. Filters:
   alignment error, palm / finger penetration in the template pose and
   contact-force balance including the object's weight. Gravity is world -z
   with the wrist at ``wrist_quat_w``, so in the object frame it differs per
   sample: ``g_obj = R_obj<-wrist g_wrist``.
2. Local refinement and validation (``GraspSynthesizer``): the grasping
   fingers are refined in simulation against the object, then the
   penetration / touch / joint-limit / balance filters and the hold test
   under gravity. The simulation runs in the wrist frame, which is the same
   physics with gravity ``g_wrist``.

The result is expressed in the wrist frame for the wrist at ``wrist_quat_w``.
An attempt is one grasp sent to step 2; at most ``k * attempts_per_sample``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

from .grasp_types import GRASP_TYPES, PAD_DEPTH, GraspType, resolve_contact
from .object_model import GraspObject
from .synth import (
  NUM_JOINTS,
  TRAINING_WRIST_QUAT_W,
  GraspSynthCfg,
  GraspSynthesizer,
  _GraspCtx,
)
from .templates import (
  TEMPLATES_PATH,
  GraspTemplate,
  HandContact,
  load_templates,
  save_templates,
  template_from_grasp,
)
from .wrench import balance_contact_forces


@dataclass
class DexonomyCfg:
  wrist_quat_w: tuple[float, float, float, float] = TRAINING_WRIST_QUAT_W
  """Wrist (palm) orientation in the world, wxyz; gravity is world -z. The
  default is the training env's wrist (palm up, pitched 10 deg)."""
  synth: GraspSynthCfg = field(default_factory=GraspSynthCfg)
  """Refinement / filters / hold test (its ``hand_quat_w`` is replaced by
  ``wrist_quat_w``)."""

  # Global alignment.
  align_samples: int = 4096
  """Hand poses sampled per round."""
  align_steps: int = 150
  align_lr_pos: float = 0.002  # m
  align_lr_rot: float = 0.03  # rad
  pos_weight: float = 1.0e4  # 1 / m^2: 1 cm costs as much as opposed normals
  normal_weight: float = 1.0
  max_align_rms: float = 0.008
  """RMS distance [m] of the contacts from their surface targets."""
  max_align_angle: float = math.radians(45.0)
  """Mean angle between hand and (opposed) object normals."""
  max_palm_penetration: float = 0.001
  max_finger_penetration: float = 0.005
  """Template-pose penetration allowed before refinement (fingers move)."""
  min_pos_gap: float = 0.005
  min_rot_gap: float = math.radians(10.0)
  """Grasps sent to refinement differ by at least this much (object pose)."""
  max_rounds: int = 10
  """Alignment rounds at most (each ``align_samples``)."""
  max_align_residual: float = 0.2
  """Balance residual allowed after alignment (refinement still has to reach
  the synth ``max_wrench_residual``)."""

  # Template update (Dexonomy): validated grasps become templates.
  grow_templates: bool = False
  """Add every validated grasp as a template of its grasp type: used from the
  next round on and kept in the synthesizer's library for later calls. The
  file is only written by ``save_templates``."""
  max_templates: int = 64
  """Templates per grasp type at most."""
  template_joint_gap: float = 0.1
  """A new template differs from every template of its type by at least
  this much [rad] on some joint."""


@dataclass
class DexonomyResult:
  """Up to k grasps; the wrist at ``wrist_quat_w``."""

  joint_names: tuple[str, ...]
  object_pose: np.ndarray
  """(n, 7) object (mesh frame) pose in the wrist frame: pos [m], quat wxyz."""
  joint_pos: np.ndarray
  """(n, 20) finger joint angles (just touching)."""
  joint_target: np.ndarray
  """(n, 20) position targets that hold the object (squeeze)."""
  wrench_residual: np.ndarray
  attempts: int
  stats: dict[str, int] = field(default_factory=dict)


def _axis_angle(v: torch.Tensor) -> torch.Tensor:
  """Rotation matrices (..., 3, 3) from axis-angle vectors (..., 3)."""
  angle = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  k = v / angle
  kx, ky, kz = k.unbind(-1)
  zero = torch.zeros_like(kx)
  skew = torch.stack((zero, -kz, ky, kz, zero, -kx, -ky, kx, zero), -1)
  skew = skew.reshape(*v.shape[:-1], 3, 3)
  s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
  eye = torch.eye(3, device=v.device).expand_as(skew)
  return eye + s * skew + (1 - c) * skew @ skew


def _rotation_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Rotations taking unit vectors a to b, (N, 3) -> (N, 3, 3)."""
  axis = torch.cross(a, b, dim=-1)
  sin = axis.norm(dim=-1, keepdim=True)
  cos = (a * b).sum(-1, keepdim=True)
  # Opposite vectors: any axis perpendicular to a.
  x_axis = torch.tensor([1.0, 0.0, 0.0], device=a.device)
  y_axis = torch.tensor([0.0, 1.0, 0.0], device=a.device)
  ref = torch.where(a[:, :1].abs() < 0.9, x_axis, y_axis)
  perp = torch.nn.functional.normalize(torch.cross(a, ref, dim=-1), dim=-1)
  axis = torch.where(sin > 1e-6, axis / sin.clamp_min(1e-9), perp)
  return _axis_angle(axis * torch.atan2(sin, cos))


def _mat_to_quat(r: torch.Tensor) -> torch.Tensor:
  """Rotation matrices (N, 3, 3) -> quaternions wxyz (N, 4), w >= 0."""
  out = np.empty((len(r), 4))
  for i, m in enumerate(r.detach().cpu().double().numpy()):
    mujoco.mju_mat2Quat(out[i], m.reshape(-1))
  out *= np.where(out[:, :1] < 0, -1.0, 1.0)
  return torch.tensor(out, dtype=torch.float32, device=r.device)


class DexonomySynthesizer:
  """``generate(obj, grasp_type, k)``: object poses and finger joints."""

  def __init__(
    self,
    cfg: DexonomyCfg | None = None,
    templates: str | Path | dict[str, list[GraspTemplate]] | None = None,
  ):
    """``templates``: the library, a templates file or a dict (default: the
    package's ``templates.yaml``)."""
    self.cfg = cfg or DexonomyCfg()
    synth_cfg = dataclasses.replace(self.cfg.synth, hand_quat_w=self.cfg.wrist_quat_w)
    self.synth = GraspSynthesizer(synth_cfg)
    if isinstance(templates, dict):
      self.library = {name: list(items) for name, items in templates.items()}
    else:
      self.library = load_templates(templates or TEMPLATES_PATH)

  def save_templates(self, path: str | Path, overwrite_default: bool = False) -> None:
    """Write the library (with the templates grown so far) to ``path``. The
    package's default file is only replaced with ``overwrite_default``."""
    if Path(path).resolve() == TEMPLATES_PATH.resolve() and not overwrite_default:
      raise ValueError(f"{path} is the default library; pass overwrite_default=True")
    save_templates(self.library, path)

  def _grow(self, key: str, source: GraspTemplate, q, pose) -> GraspTemplate | None:
    """Template from a validated grasp, if it touches with the source's
    fingers and differs from the type's templates; added to the library."""
    cfg = self.cfg
    items = self.library.setdefault(key, [])
    if len(items) >= cfg.max_templates:
      return None
    tpl = template_from_grasp(
      self.synth.model, key, q.cpu().numpy(), pose.cpu().numpy(),
      self.synth.cfg.touch_dist,
    )  # fmt: skip
    if not {c.finger for c in source.contacts} <= {c.finger for c in tpl.contacts}:
      return None
    joints = np.asarray(tpl.joint_pos)
    for other in items:
      if np.abs(joints - np.asarray(other.joint_pos)).max() < cfg.template_joint_gap:
        return None
    items.append(tpl)
    return tpl

  def _template(self, q: torch.Tensor, ctx) -> tuple[torch.Tensor, torch.Tensor]:
    """Grasp-type contact points / outward normals in the wrist frame."""
    m = self.synth.model
    d = mujoco.MjData(m)
    d.qpos[:NUM_JOINTS] = q.cpu().numpy()
    mujoco.mj_kinematics(m, d)
    body = ctx.body.cpu().numpy()
    rot = torch.tensor(d.xmat[body].reshape(-1, 3, 3), dtype=torch.float32)
    pos = torch.tensor(d.xpos[body], dtype=torch.float32)
    dev = ctx.body.device
    rot, pos = rot.to(dev), pos.to(dev)
    p = pos + torch.einsum("cij,cj->ci", rot, ctx.point)
    return p, torch.einsum("cij,cj->ci", rot, ctx.normal)

  def _align(self, lib, gen):
    """Global alignment of ``align_samples`` hand poses spread over the
    templates in ``lib`` (padded contacts), object frame, x_obj = R x_wrist + t.
    Returns the template of each sample, R (N, 3, 3), t (N, 3), per-contact
    distance to the surface target and normal angle (N, C), and the nearest
    surface points / normals."""
    cfg, dev = self.cfg, self.synth.cfg.device
    n = cfg.align_samples
    surface = self.synth.surface
    pts, nrm = self._surface_pts, self._surface_nrm
    tpl = torch.randint(len(lib["q"]), (n,), device=dev, generator=gen)
    p_h, n_h, mask = lib["p"][tpl], lib["n"][tpl], lib["mask"][tpl]
    i = torch.multinomial(mask.float(), 1, generator=gen)[:, 0]
    rows = torch.arange(n, device=dev)
    s = torch.randint(len(pts), (n,), device=dev, generator=gen)
    theta = torch.rand(n, device=dev, generator=gen) * 2 * math.pi
    r0 = _rotation_between(n_h[rows, i], -nrm[s])
    r0 = _axis_angle(nrm[s] * theta[:, None]) @ r0
    t0 = pts[s] + PAD_DEPTH * nrm[s] - torch.einsum("nij,nj->ni", r0, p_h[rows, i])

    w = torch.zeros(n, 3, device=dev, requires_grad=True)
    dt = torch.zeros(n, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam(
      [
        {"params": [w], "lr": cfg.align_lr_rot},
        {"params": [dt], "lr": cfg.align_lr_pos},
      ]
    )
    weight = mask.float() / mask.sum(1, keepdim=True)
    for step in range(cfg.align_steps + 1):
      r = _axis_angle(w) @ r0
      p = torch.einsum("nij,ncj->nci", r, p_h) + (t0 + dt)[:, None]
      nh = torch.einsum("nij,ncj->nci", r, n_h)
      with torch.no_grad():
        near, normal, _ = surface.closest(p)
      target = near + PAD_DEPTH * normal
      err_p = ((p - target) ** 2).sum(-1)
      err_n = ((nh + normal) ** 2).sum(-1)
      energy = (weight * (cfg.pos_weight * err_p + cfg.normal_weight * err_n)).sum(1)
      if step == cfg.align_steps:
        break
      opt.zero_grad()
      energy.sum().backward()
      opt.step()
    with torch.no_grad():
      cos = -(nh * normal).sum(-1).clamp(-1.0, 1.0)
      dist = err_p.sqrt().masked_fill(~mask, 0.0)
      angle = torch.acos(cos).masked_fill(~mask, 0.0)
      return tpl, r.detach(), (t0 + dt).detach(), dist, angle, near, normal

  def _batch_ctx(self, prepared) -> _GraspCtx:
    """The templates' contexts stacked, one row per template, contacts
    padded (``mask``)."""
    syn, dev = self.synth, self.synth.cfg.device
    t, c = len(prepared), max(len(x[2]) for x in prepared)
    body = torch.zeros(t, c, dtype=torch.long, device=dev)
    point = torch.zeros(t, c, 3, device=dev)
    normal = torch.zeros(t, c, 3, device=dev)
    normal[..., 2] = 1.0
    finger = torch.zeros(t, c, dtype=torch.long, device=dev)
    mask = torch.zeros(t, c, dtype=torch.bool, device=dev)
    fingers = torch.zeros(t, 6, dtype=torch.bool, device=dev)
    for k, (ctx, *_) in enumerate(prepared):
      m = len(ctx.body)
      body[k, :m], point[k, :m], normal[k, :m] = ctx.body, ctx.point, ctx.normal
      finger[k, :m], mask[k, :m] = ctx.finger, True
      fingers[k] = ctx.finger_mask()
    return _GraspCtx(
      body=body,
      point=point,
      normal=normal,
      chain=syn.body_chain[body] * mask[..., None],
      finger=finger,
      fingers=fingers,
      active_joint=fingers[:, syn.joint_finger],
      mask=mask,
    )

  def _library_tensors(self, prepared) -> dict[str, torch.Tensor]:
    """Template contacts padded to the most contacts: points, normals (wrist
    frame), mask, finger, joint angles."""
    dev = self.synth.cfg.device
    t, c = len(prepared), max(len(x[2]) for x in prepared)
    out = {
      "p": torch.zeros(t, c, 3, device=dev),
      "n": torch.zeros(t, c, 3, device=dev),
      "mask": torch.zeros(t, c, dtype=torch.bool, device=dev),
      "finger": torch.zeros(t, c, dtype=torch.long, device=dev),
      "q": torch.stack([x[1] for x in prepared]),
    }
    for k, (ctx, _, p_h, n_h) in enumerate(prepared):
      m = len(p_h)
      out["p"][k, :m], out["n"][k, :m] = p_h, n_h
      out["mask"][k, :m] = True
      out["finger"][k, :m] = ctx.finger
    out["n"][~out["mask"]] = torch.tensor([0.0, 0.0, 1.0], device=dev)
    return out

  def _penetration(self, pose, q):
    """Deepest palm / finger penetration [m] of hands (B, 20) at object poses
    (B, 7) in the wrist frame, before refinement."""
    syn = self.synth

    def run(pose, q):
      data = syn.sim.data
      syn.margin.copy_(syn.margin0)
      syn.gap.copy_(syn.gap0)
      data.qpos[:, :NUM_JOINTS] = q
      syn._pin_object(pose)
      syn.sim.forward()
      n = int(data.nacon[0])
      world, geom = data.contact.worldid[:n].long(), data.contact.geom[:n].long()
      dist = data.contact.dist[:n]
      obj1 = syn.obj_geom[geom[:, 0]]
      hand_obj = obj1 ^ syn.obj_geom[geom[:, 1]]
      finger = syn.geom_finger[torch.where(obj1, geom[:, 1], geom[:, 0])]
      group = torch.where(hand_obj, (finger > 0).long(), 2)  # 0 palm, 1 finger
      depth = syn._max_per(world, group, -dist, syn.nworld, 3)
      return {"palm": depth[:, 0], "finger": depth[:, 1]}

    return syn._chunked(run, pose, q)

  def _templates(self, grasp_type, hand_q) -> list[GraspTemplate]:
    if isinstance(grasp_type, GraspTemplate):
      return [grasp_type]
    if isinstance(grasp_type, (list, tuple)):
      return list(grasp_type)
    if hand_q is None:
      return list(self.library[grasp_type])
    # A given hand shape: the grasp type's pad contacts.
    gtype = GRASP_TYPES[grasp_type] if isinstance(grasp_type, str) else grasp_type
    model, contacts = self.synth.model, []
    for spec in gtype.contacts:
      r = resolve_contact(model, spec)
      surface = r.point + PAD_DEPTH * r.normal
      contacts.append(
        HandContact(model.body(r.body_id).name, tuple(surface), tuple(r.normal))
      )
    if len(hand_q) != NUM_JOINTS:
      raise ValueError(f"hand_q needs {NUM_JOINTS} joint angles")
    return [GraspTemplate(gtype.name, tuple(hand_q), tuple(contacts))]

  def _ctx(self, tpl: GraspTemplate) -> _GraspCtx:
    """Template contacts as grasp tensors; reference points PAD_DEPTH inside
    the surface, as for the grasp-type pads."""
    syn, dev = self.synth, self.synth.cfg.device
    body = torch.tensor([syn.model.body(c.body).id for c in tpl.contacts], device=dev)
    normal = torch.tensor([c.normal for c in tpl.contacts], device=dev).float()
    point = torch.tensor([c.point for c in tpl.contacts], device=dev).float()
    # Grasping fingers, and the palm when the template touches with it.
    fingers = torch.tensor(sorted({c.finger for c in tpl.contacts}), device=dev)
    return _GraspCtx(
      body=body,
      point=point - PAD_DEPTH * normal,
      normal=normal,
      chain=syn.body_chain[body],
      finger=torch.tensor([c.finger for c in tpl.contacts], device=dev),
      fingers=fingers,
      active_joint=torch.isin(syn.joint_finger, fingers),
    )

  def generate(
    self,
    obj: GraspObject,
    grasp_type: str | GraspType | GraspTemplate | Sequence[GraspTemplate],
    k: int,
    hand_q: Sequence[float] | None = None,
  ) -> DexonomyResult:
    """k grasps of ``obj``. ``grasp_type``: a library template name (all its
    templates are used), a ``GraspTemplate`` or a list of them; with
    ``hand_q`` (20 joint angles) the hand shape is given and the contacts are
    the grasp type's pads."""
    cfg, scfg = self.cfg, self.synth.cfg
    syn, dev = self.synth, scfg.device
    budget = k * scfg.attempts_per_sample
    syn._setup(obj, min(scfg.num_worlds, max(cfg.align_samples, budget)))
    templates = self._templates(grasp_type, hand_q)
    key = templates[0].name

    def prepare(tpl: GraspTemplate):
      ctx = self._ctx(tpl)
      q = torch.tensor(tpl.joint_pos, dtype=torch.float32, device=dev)
      return (ctx, q, *self._template(q, ctx))

    prepared = [prepare(tpl) for tpl in templates]
    templates_used = list(templates)  # parallel to prepared
    merged = trimesh.util.concatenate(obj.collision_parts)
    pts, face = trimesh.sample.sample_surface(merged, 20000, seed=scfg.seed)
    self._surface_pts = torch.tensor(pts, dtype=torch.float32, device=dev)
    self._surface_nrm = torch.tensor(
      merged.face_normals[face], dtype=torch.float32, device=dev
    )
    gen = torch.Generator(device=dev).manual_seed(scfg.seed)
    com = torch.tensor(obj.com, dtype=torch.float32, device=dev)

    found: list[tuple] = []
    attempts = 0
    stats = dict.fromkeys(
      ("sampled", "aligned", "no_pen", "balanced", "distinct", "verified",
       "refined_no_pen", "touch", "limits", "wrench", "hold", "new_templates"), 0,
    )  # fmt: skip
    rounds = 0
    while len(found) < k and attempts < budget and rounds < cfg.max_rounds:
      rounds += 1
      need = k - len(found)
      # 1. Global alignment in the object frame, all templates at once.
      lib = self._library_tensors(prepared)
      owner, r, t, dist, angle, near, normal = self._align(lib, gen)
      mask = lib["mask"][owner]
      stats["sampled"] += len(r)
      count = mask.sum(1)
      rms = (dist.pow(2).sum(1) / count).sqrt()
      ok = (rms <= cfg.max_align_rms) & (angle.sum(1) / count <= cfg.max_align_angle)
      stats["aligned"] += int(ok.sum())
      idx = torch.nonzero(ok).flatten()
      poses = residuals = None
      if len(idx):
        # Object pose in the wrist frame: R^T, -R^T t.
        rot_w = r[idx].transpose(1, 2)
        pose = torch.cat(
          (-torch.einsum("nij,nj->ni", rot_w, t[idx]), _mat_to_quat(rot_w)), 1
        )
        pen = self._penetration(pose, lib["q"][owner[idx]])
        clear = (pen["palm"] <= cfg.max_palm_penetration) & (
          pen["finger"] <= cfg.max_finger_penetration
        )
        idx, pose = idx[clear], pose[clear]
        stats["no_pen"] += len(idx)
      if len(idx):
        # Balance in the object frame; gravity there depends on the sample.
        g_obj = torch.einsum("nij,j->ni", r[idx], syn.gravity_palm)
        goal = torch.zeros(len(idx), 6, device=dev)
        goal[:, :3] = -obj.mass * g_obj / scfg.grip_force
        # Fingers close onto the surface in refinement; the palm does not move.
        finger = lib["finger"][owner[idx]]
        in_contact = mask[idx] & ((finger > 0) | (dist[idx] <= scfg.contact_tol))
        residual, _ = balance_contact_forces(
          near[idx], -normal[idx], in_contact, com.expand(len(idx), -1), obj.radius,
          scfg.friction, min_share=scfg.min_force_share, target=goal,
        )  # fmt: skip
        keep = residual <= cfg.max_align_residual
        stats["balanced"] += int(keep.sum())
        poses, residuals, owner = pose[keep], residual[keep], owner[idx][keep]
      limit = min(budget - attempts, need * scfg.verify_oversample)
      if poses is not None and len(poses):
        idx = self._distinct(poses, residuals, found, limit)
      else:
        idx = torch.zeros(0, dtype=torch.long, device=dev)
      stats["distinct"] += len(idx)
      attempts += len(idx)
      if len(idx) == 0:
        continue

      # 2. Local refinement and validation (wrist frame), all at once.
      tpl = owner[idx]
      ctx = self._batch_ctx(prepared).rows(tpl)
      res = syn._chunked(syn._verify, poses[idx], lib["q"][tpl], ctx)
      stats["verified"] += len(idx)
      stats["refined_no_pen"] += int(res["no_pen"].sum())
      for s in ("touch", "limits"):
        stats[s] += int(res[s].sum())
      stats["wrench"] += int(res["balanced"].sum())
      stats["hold"] += int(res["ok"].sum())
      for w in torch.nonzero(res["ok"]).flatten().tolist():
        pose = poses[idx[w]]
        if len(found) < k:
          found.append((pose, res["q"][w], res["q_target"][w], res["residual"][w]))
        if cfg.grow_templates:
          new = self._grow(key, templates_used[int(tpl[w])], res["q"][w], pose)
          if new is not None:
            templates_used.append(new)
            prepared.append(prepare(new))
            stats["new_templates"] += 1

    def stack(i, width):
      if not found:
        return np.zeros((0, *width), dtype=np.float32)
      return torch.stack([f[i] for f in found]).cpu().numpy().reshape(-1, *width)

    return DexonomyResult(
      joint_names=syn.joint_names,
      object_pose=stack(0, (7,)),
      joint_pos=stack(1, (NUM_JOINTS,)),
      joint_target=stack(2, (NUM_JOINTS,)),
      wrench_residual=stack(3, ()),
      attempts=int(attempts),
      stats=stats,
    )

  def _distinct(self, pose, residual, found, limit: int) -> torch.Tensor:
    """Greedy, best residual first: indices of poses that differ from every
    kept one and every grasp found by ``min_pos_gap`` or ``min_rot_gap``."""
    cfg = self.cfg
    kept = [f[0] for f in found]
    out = []
    for i in torch.argsort(residual).tolist():
      p = pose[i]
      if kept:
        ref = torch.stack(kept)
        close = ((ref[:, :3] - p[:3]).norm(dim=-1) < cfg.min_pos_gap) & (
          2 * torch.acos((ref[:, 3:] @ p[3:]).abs().clamp(max=1.0)) < cfg.min_rot_gap
        )
        if close.any():
          continue
      kept.append(p)
      out.append(i)
      if len(out) == limit:
        break
    return torch.tensor(out, dtype=torch.long, device=pose.device)
