# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""The object to grasp: mesh, collision parts, mass and surface queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp


@dataclass
class GraspObject:
  """A rigid object in its own (mesh) frame; poses place this frame.

  ``collision_parts`` are convex parts (mujoco-warp collides convex hulls
  only); by default the convex hull of ``mesh``, which fills concavities. Pass
  a convex decomposition for concave objects. Nearest-point targets and
  contact checks use the same parts.
  """

  mesh: trimesh.Trimesh
  mass: float = 0.1
  collision_parts: list[trimesh.Trimesh] = field(default_factory=list)
  max_hull_vertices: int = 64

  def __post_init__(self) -> None:
    if not self.collision_parts:
      self.collision_parts = [self.mesh.convex_hull]
    self.collision_parts = [p.convex_hull for p in self.collision_parts]

  @classmethod
  def from_file(
    cls,
    path: str | Path,
    mass: float = 0.1,
    collision: list[str | Path] | None = None,
    scale: float = 1.0,
  ) -> GraspObject:
    def load(p) -> trimesh.Trimesh:
      mesh = trimesh.load(str(p), force="mesh")
      mesh.apply_scale(scale)
      return mesh

    parts = [load(p) for p in collision] if collision else []
    return cls(load(path), mass, parts)

  @classmethod
  def box(cls, half_size, mass: float = 0.1) -> GraspObject:
    return cls(trimesh.creation.box(extents=2 * np.asarray(half_size)), mass)

  @classmethod
  def from_compound(cls, name: str, mass: float | None = None) -> GraspObject:
    """A repo compound object (``compound_objects.COMPOUND_OBJECTS``): one
    convex collision part per primitive, in the parts' frame; the mass
    defaults to the parts' densities."""
    from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
      COMPOUND_OBJECTS,
    )

    parts = []
    for part in COMPOUND_OBJECTS[name]():
      if part.kind == "box":
        mesh = trimesh.creation.box(extents=2 * np.asarray(part.size))
      else:
        radius, half_length = part.size[:2]
        mesh = trimesh.creation.cylinder(radius, 2 * half_length, sections=32)
      pose = trimesh.transformations.quaternion_matrix(part.quat)
      pose[:3, 3] = part.pos
      parts.append(mesh.apply_transform(pose))
    if mass is None:
      mass = sum(p.density * p.volume for p in COMPOUND_OBJECTS[name]())
    return cls(trimesh.util.concatenate(parts), mass, parts)

  @property
  def com(self) -> np.ndarray:
    hull = trimesh.util.concatenate(self.collision_parts)
    return np.asarray(
      self.mesh.center_mass if self.mesh.is_watertight else hull.center_mass
    )

  @property
  def inertia(self) -> np.ndarray:
    """3x3 inertia about the COM for ``mass``, mesh frame axes."""
    solid = self.mesh if self.mesh.is_watertight else self.mesh.convex_hull
    return np.asarray(solid.moment_inertia) * (self.mass / solid.mass)

  @property
  def radius(self) -> float:
    """Largest distance from the COM to the collision surface."""
    verts = np.concatenate([p.vertices for p in self.collision_parts])
    return float(np.linalg.norm(verts - self.com, axis=1).max())


@wp.kernel
def _closest_kernel(
  mesh: wp.uint64,
  points: wp.array(dtype=wp.vec3),
  max_dist: float,
  out_point: wp.array(dtype=wp.vec3),
  out_normal: wp.array(dtype=wp.vec3),
  out_dist: wp.array(dtype=float),
):
  i = wp.tid()
  p = points[i]
  q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
  if q.result:
    c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    out_point[i] = c
    out_normal[i] = wp.mesh_eval_face_normal(mesh, q.face)
    out_dist[i] = wp.length(p - c) * q.sign
  else:
    out_point[i] = p
    out_normal[i] = wp.vec3(0.0, 0.0, 1.0)
    out_dist[i] = max_dist


class SurfaceQuery:
  """Closest points on the collision parts (object frame), batched on device."""

  def __init__(self, obj: GraspObject, device: str):
    merged = trimesh.util.concatenate(obj.collision_parts)
    self.device = device
    self._mesh = wp.Mesh(
      points=wp.array(merged.vertices.astype(np.float32), dtype=wp.vec3, device=device),
      indices=wp.array(
        merged.faces.reshape(-1).astype(np.int32), dtype=int, device=device
      ),
    )

  def closest(
    self, points: torch.Tensor, max_dist: float = 1.0
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``points`` (..., 3), object frame -> closest point (..., 3), outward
    face normal (..., 3) and signed distance (...,), negative inside."""
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3).float().contiguous()
    n = flat.shape[0]
    out_p = torch.empty_like(flat)
    out_n = torch.empty_like(flat)
    out_d = torch.empty(n, device=flat.device)
    wp.launch(
      _closest_kernel,
      dim=n,
      inputs=[self._mesh.id, wp.from_torch(flat, dtype=wp.vec3), max_dist],
      outputs=[
        wp.from_torch(out_p, dtype=wp.vec3),
        wp.from_torch(out_n, dtype=wp.vec3),
        wp.from_torch(out_d),
      ],
      device=self.device,
    )
    return out_p.reshape(*shape, 3), out_n.reshape(*shape, 3), out_d.reshape(shape)
