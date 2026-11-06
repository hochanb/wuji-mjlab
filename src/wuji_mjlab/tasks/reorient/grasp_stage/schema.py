# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Training / stage / object / grasp-set files (YAML) and their validation.

A training file lists stages; each stage names its objects and, per object,
the grasp sets used as initial states and as goals. Objects live in
``<object_root>/<name>/object.yaml``; a grasp set is a folder holding
``grasps.yaml``, given relative to its object folder. Everything is checked
before a stage env is built (``load_training``).

Grasp scheme ``xi = {T_wrist^object, q_tip, q_joint, w_tip, w_joint}``:
- ``object_pose``: object pose in the hand base (wrist) frame [m, wxyz], with
  optional ``weights: {pos, rot}`` in [0, 1] (default 1; 0 = not required).
- ``tip``: per finger (thumb to pinky) a target point on the object surface
  and its outward normal, object frame; ``weights`` in [0, 1] (0 = off).
- ``joint``: finger joint angles (wrist excluded) and per-finger ``weights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

SCHEMA_VERSION = 1


class GraspStageError(ValueError):
  """A training, object or grasp file failed validation."""


@dataclass(frozen=True)
class HandSpec:
  fingers: int
  joints_per_finger: int
  # Depth of the fingertip reference point inside the pad [m]: a pad touching
  # its target puts the reference point this far out along the normal.
  pad_depth: float

  @property
  def finger_joints(self) -> int:
    return self.fingers * self.joints_per_finger


HANDS = {"wuji_hand2_right": HandSpec(fingers=5, joints_per_finger=4, pad_depth=0.005)}


@dataclass(frozen=True)
class Grasp:
  id: str
  object_pos: tuple[float, ...]  # (3,) hand base frame [m]
  object_quat: tuple[float, ...]  # (4,) wxyz, unit
  pose_weights: tuple[float, float]  # (w_pos, w_rot)
  tip_points: tuple[tuple[float, ...], ...]  # (F, 3) object frame, on the surface
  tip_normals: tuple[tuple[float, ...], ...]  # (F, 3) unit, outward
  tip_weights: tuple[float, ...]  # (F,)
  joint_q: tuple[float, ...]  # (F * J,) finger joints, thumb to pinky
  joint_weights: tuple[float, ...]  # (F,)


@dataclass(frozen=True)
class ObjectSpec:
  name: str
  kind: str  # "compound" | "mesh"
  folder: Path
  compound: str | None = None
  mesh_visual: Path | None = None
  mesh_collision: tuple[Path, ...] = ()
  encoding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class StageSpec:
  name: str
  objects: tuple[ObjectSpec, ...]
  init_grasps: dict[str, tuple[Grasp, ...]]  # object name -> merged grasps
  goal_grasps: dict[str, tuple[Grasp, ...]]
  advance_after_iterations: int


@dataclass(frozen=True)
class TrainingSpec:
  path: Path
  hand: str
  encoding_name: str
  encoding_dim: int
  stages: tuple[StageSpec, ...]

  @property
  def hand_spec(self) -> HandSpec:
    return HANDS[self.hand]

  @property
  def total_iterations(self) -> int:
    return sum(s.advance_after_iterations for s in self.stages)


def _read_yaml(path: Path) -> dict:
  if not path.is_file():
    raise GraspStageError(f"{path}: file not found")
  with open(path) as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise GraspStageError(f"{path}: expected a mapping at the top level")
  return data


def _require(data: dict, key: str, where: str):
  if key not in data:
    raise GraspStageError(f"{where}: missing '{key}'")
  return data[key]


def _vector(value, length: int, where: str) -> np.ndarray:
  arr = np.asarray(value, dtype=float)
  if arr.shape != (length,) or not np.all(np.isfinite(arr)):
    raise GraspStageError(f"{where}: expected {length} finite numbers, got {value!r}")
  return arr


def _matrix(value, rows: int, cols: int, where: str) -> np.ndarray:
  arr = np.asarray(value, dtype=float)
  if arr.shape != (rows, cols) or not np.all(np.isfinite(arr)):
    raise GraspStageError(f"{where}: expected {rows} x {cols} finite numbers")
  return arr


def _unit_rows(arr: np.ndarray, where: str, tol: float = 1e-3) -> np.ndarray:
  norms = np.linalg.norm(arr, axis=-1)
  if np.any(np.abs(norms - 1.0) > tol):
    raise GraspStageError(f"{where}: not unit length (norms {np.round(norms, 4)})")
  return arr / norms[..., None]


def _weights(value, length: int, where: str) -> np.ndarray:
  arr = _vector(value, length, where)
  if np.any(arr < 0.0) or np.any(arr > 1.0):
    raise GraspStageError(f"{where}: weights must lie in [0, 1], got {arr}")
  return arr


def _tuple(arr: np.ndarray):
  return tuple(map(_tuple, arr)) if arr.ndim > 1 else tuple(float(x) for x in arr)


def load_grasp_set(folder: Path, hand: str) -> tuple[Grasp, ...]:
  path = folder / "grasps.yaml"
  data = _read_yaml(path)
  where = str(path)
  if _require(data, "version", where) != SCHEMA_VERSION:
    raise GraspStageError(f"{where}: version must be {SCHEMA_VERSION}")
  if _require(data, "hand", where) != hand:
    raise GraspStageError(f"{where}: hand {data['hand']!r}, training uses {hand!r}")
  if _require(data, "units", where) != {"length": "m", "angle": "rad"}:
    raise GraspStageError(f"{where}: units must be {{length: m, angle: rad}}")
  if _require(data, "frames", where) != {"object_pose": "wrist", "tip": "object"}:
    raise GraspStageError(
      f"{where}: frames must be {{object_pose: wrist, tip: object}}"
    )
  spec = HANDS[hand]
  raw = _require(data, "grasps", where)
  if not isinstance(raw, list) or not raw:
    raise GraspStageError(f"{where}: 'grasps' must be a non-empty list")
  grasps, ids = [], set()
  for k, g in enumerate(raw):
    w = f"{where} grasps[{k}]"
    gid = str(_require(g, "id", w))
    if gid in ids:
      raise GraspStageError(f"{w}: duplicate id {gid!r}")
    ids.add(gid)
    pose, tip, joint = (_require(g, key, w) for key in ("object_pose", "tip", "joint"))
    pos = _vector(_require(pose, "pos", w + ".object_pose"), 3, w + ".object_pose.pos")
    quat = _vector(
      _require(pose, "quat_wxyz", w + ".object_pose"), 4, w + ".object_pose.quat_wxyz"
    )
    quat = _unit_rows(quat[None], w + ".object_pose.quat_wxyz")[0]
    pw = pose.get("weights", {"pos": 1.0, "rot": 1.0})
    if not isinstance(pw, dict) or set(pw) != {"pos", "rot"}:
      raise GraspStageError(f"{w}.object_pose.weights: needs exactly pos and rot")
    pose_w = _weights([pw["pos"], pw["rot"]], 2, w + ".object_pose.weights")
    points = _matrix(
      _require(tip, "points", w + ".tip"), spec.fingers, 3, w + ".tip.points"
    )
    normals = _matrix(
      _require(tip, "normals", w + ".tip"), spec.fingers, 3, w + ".tip.normals"
    )
    normals = _unit_rows(normals, w + ".tip.normals")
    tip_w = _weights(
      _require(tip, "weights", w + ".tip"), spec.fingers, w + ".tip.weights"
    )
    q = _vector(_require(joint, "q", w + ".joint"), spec.finger_joints, w + ".joint.q")
    joint_w = _weights(
      _require(joint, "weights", w + ".joint"), spec.fingers, w + ".joint.weights"
    )
    grasps.append(
      Grasp(
        gid, _tuple(pos), _tuple(quat), _tuple(pose_w), _tuple(points), _tuple(normals),
        _tuple(tip_w), _tuple(q), _tuple(joint_w),
      )
    )  # fmt: skip
  return tuple(grasps)


def load_object(folder: Path, encoding_dim: int) -> ObjectSpec:
  path = folder / "object.yaml"
  data = _read_yaml(path)
  where = str(path)
  name = _require(data, "name", where)
  if name != folder.name:
    raise GraspStageError(f"{where}: name {name!r} must match its folder")
  kind = _require(data, "kind", where)
  compound, visual, collision = None, None, ()
  if kind == "compound":
    from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
      COMPOUND_OBJECTS,
    )

    compound = _require(data, "compound", where)
    if compound not in COMPOUND_OBJECTS:
      raise GraspStageError(
        f"{where}: unknown compound builder {compound!r} "
        f"(known: {sorted(COMPOUND_OBJECTS)})"
      )
  elif kind == "mesh":
    mesh = _require(data, "mesh", where)
    visual = folder / _require(mesh, "visual", where + ".mesh")
    collision = tuple(folder / p for p in _require(mesh, "collision", where + ".mesh"))
    for p in (visual, *collision):
      if not p.is_file():
        raise GraspStageError(f"{where}: mesh file {p} not found")
    if not collision:
      raise GraspStageError(f"{where}: mesh.collision needs at least one convex part")
  else:
    raise GraspStageError(f"{where}: kind must be 'compound' or 'mesh', got {kind!r}")
  encoding = None
  if "encoding" in data:
    enc = np.load(folder / data["encoding"]).astype(float).ravel()
    if enc.shape != (encoding_dim,):
      raise GraspStageError(
        f"{where}: encoding has {enc.size} values, training uses {encoding_dim}"
      )
    encoding = _tuple(enc)
  elif encoding_dim > 0:
    raise GraspStageError(f"{where}: training needs a {encoding_dim}-d encoding")
  return ObjectSpec(name, kind, folder, compound, visual, collision, encoding)


def load_training(path: str | Path) -> TrainingSpec:
  """Load and validate a training file and everything it references."""
  path = Path(path).resolve()
  data = _read_yaml(path)
  where = str(path)
  if _require(data, "version", where) != SCHEMA_VERSION:
    raise GraspStageError(f"{where}: version must be {SCHEMA_VERSION}")
  hand = _require(data, "hand", where)
  if hand not in HANDS:
    raise GraspStageError(f"{where}: unknown hand {hand!r} (known: {sorted(HANDS)})")
  encoding = data.get("encoding", {"name": "none", "dim": 0})
  enc_name, enc_dim = str(encoding.get("name", "none")), int(encoding.get("dim", 0))
  if enc_dim < 0:
    raise GraspStageError(f"{where}: encoding.dim must be >= 0")
  object_root = path.parent / data.get("object_root", "objects")
  raw_stages = _require(data, "stages", where)
  if not isinstance(raw_stages, list) or not raw_stages:
    raise GraspStageError(f"{where}: 'stages' must be a non-empty list")

  objects: dict[str, ObjectSpec] = {}
  stages, names = [], set()
  for k, s in enumerate(raw_stages):
    w = f"{where} stages[{k}]"
    name = str(_require(s, "name", w))
    if name in names:
      raise GraspStageError(f"{w}: duplicate stage name {name!r}")
    names.add(name)
    obj_names = _require(s, "objects", w)
    if not isinstance(obj_names, list) or not obj_names:
      raise GraspStageError(f"{w}: 'objects' must be a non-empty list")
    if len(obj_names) > 1:
      raise GraspStageError(f"{w}: stages with several objects are not supported yet")
    for n in obj_names:
      if n not in objects:
        objects[n] = load_object(object_root / n, enc_dim)
    stage_objects = tuple(objects[n] for n in obj_names)
    if len({o.kind for o in stage_objects}) > 1:
      raise GraspStageError(f"{w}: mesh and compound objects cannot share a stage")
    sets = {}
    for key in ("init_grasps", "goal_grasps"):
      per_object = _require(s, key, w)
      if set(per_object) != set(obj_names):
        raise GraspStageError(
          f"{w}.{key}: needs exactly the stage objects {sorted(obj_names)}, "
          f"got {sorted(per_object)}"
        )
      merged = {}
      for n, folders in per_object.items():
        if not folders:
          raise GraspStageError(f"{w}.{key}.{n}: list at least one grasp set")
        grasps = []
        for rel in folders:
          grasps.extend(load_grasp_set(objects[n].folder / rel, hand))
        merged[n] = tuple(grasps)
      sets[key] = merged
    advance = _require(s, "advance", w)
    iters = advance.get("after_iterations") if isinstance(advance, dict) else None
    if not isinstance(iters, int) or iters <= 0:
      raise GraspStageError(f"{w}.advance: after_iterations must be a positive int")
    stages.append(
      StageSpec(name, stage_objects, sets["init_grasps"], sets["goal_grasps"], iters)
    )
  return TrainingSpec(path, hand, enc_name, enc_dim, tuple(stages))
