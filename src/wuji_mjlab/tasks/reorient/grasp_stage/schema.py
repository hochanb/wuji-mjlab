# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Training / stage / object / grasp-set files and their validation.

A training file (YAML) lists stages; each stage names its objects and, per
object, the grasp sets used as initial states and as goals. Objects live in
``<object_root>/<name>/object.yaml``; a grasp set is a folder, given relative
to its object folder, holding either ``grasps.yaml`` (small, hand-edited) or
``grasps.npz`` (large, generated; see ``save_grasp_set_npz``). Everything is
checked before a stage env is built (``load_training``).

Grasp scheme ``xi = {T_wrist^object, q_tip, q_joint, w_tip, w_joint}``:
- ``object_pose``: object pose in the hand base (wrist) frame [m, wxyz], with
  optional ``weights: {pos, rot}`` in [0, 1] (default 1; 0 = not required).
- ``tip``: per finger (thumb to pinky) a target point on the object surface
  and its outward normal, object frame; ``weights`` in [0, 1] (0 = off).
- ``joint``: finger joint angles (wrist excluded) and per-finger ``weights``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import yaml

SCHEMA_VERSION = 1
_UNITS = {"length": "m", "angle": "rad"}
_FRAMES = {"object_pose": "wrist", "tip": "object"}
_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # C parser when built


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


@cache
def hand_joint_limits(hand: str) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
  """Finger joint names (grasp order) with their range and soft limits
  [(F*J, 2) rad], from the hand model the env uses (compiled once)."""
  if hand != "wuji_hand2_right":
    raise GraspStageError(f"no joint limits for hand {hand!r}")
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.env_cfgs import (
    get_wuji_hand2_rig_cfg,
  )

  robot = get_wuji_hand2_rig_cfg("right")
  model = robot.spec_fn().compile()
  spec = HANDS[hand]
  names = tuple(
    f"right_finger{f}_joint{j}"
    for f in range(1, spec.fingers + 1)
    for j in range(1, spec.joints_per_finger + 1)
  )
  hard = np.array([model.joint(n).range for n in names], dtype=np.float64)
  factor = robot.articulation.soft_joint_pos_limit_factor
  center, half = hard.mean(-1), 0.5 * (hard[:, 1] - hard[:, 0])
  soft = np.stack((center - factor * half, center + factor * half), -1)
  return names, hard, soft


def _check_joint_limits(grasps: GraspSet, hand: str, role: str, where: str) -> None:
  """Initial grasps: every joint within its range (the hand is put there).
  Goal grasps: joints of fingers with w_joint > 0 within the soft limits the
  env's joint targets use, so the goal is reachable."""
  names, hard, soft = hand_joint_limits(hand)
  lo, hi = (hard if role == "init" else soft).T
  q = grasps.joint_q.astype(np.float64)
  out = (q < lo - 1e-6) | (q > hi + 1e-6)
  if role != "init":
    per_finger = HANDS[hand].joints_per_finger
    out &= np.repeat(grasps.joint_weights > 0, per_finger, axis=1)
  if out.any():
    g, j = np.argwhere(out)[0]
    kind = "range" if role == "init" else "soft limits"
    raise GraspStageError(
      f"{where}: grasp id {grasps.ids[g]} {names[j]} = {q[g, j]:.4f} rad is outside "
      f"its {kind} [{lo[j]:.4f}, {hi[j]:.4f}]"
    )


@dataclass(frozen=True, eq=False)
class GraspSet:
  """G grasps as arrays (float32), one row per grasp."""

  ids: np.ndarray  # (G,) int64, unique within a set (default: row index)
  object_pos: np.ndarray  # (G, 3) hand base frame [m]
  object_quat: np.ndarray  # (G, 4) wxyz, unit
  pose_weights: np.ndarray  # (G, 2) (w_pos, w_rot)
  tip_points: np.ndarray  # (G, F, 3) object frame, on the surface
  tip_normals: np.ndarray  # (G, F, 3) unit, outward
  tip_weights: np.ndarray  # (G, F)
  joint_q: np.ndarray  # (G, F * J) finger joints, thumb to pinky
  joint_weights: np.ndarray  # (G, F)

  ARRAYS = (
    "object_pos",
    "object_quat",
    "pose_weights",
    "tip_points",
    "tip_normals",
    "tip_weights",
    "joint_q",
    "joint_weights",
  )

  def __len__(self) -> int:
    return len(self.ids)

  @staticmethod
  def concat(sets: list[GraspSet]) -> GraspSet:
    return GraspSet(
      np.concatenate([s.ids for s in sets]),
      *(np.concatenate([getattr(s, a) for s in sets]) for a in GraspSet.ARRAYS),
    )


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
  init_grasps: dict[str, GraspSet]  # object name -> merged grasp sets
  goal_grasps: dict[str, GraspSet]
  advance_after_iterations: int
  # Grasp-set folders behind init_grasps / goal_grasps, per object.
  init_set_folders: dict[str, tuple[Path, ...]] = field(default_factory=dict)
  goal_set_folders: dict[str, tuple[Path, ...]] = field(default_factory=dict)


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
    data = yaml.load(f, Loader=_Loader)
  if not isinstance(data, dict):
    raise GraspStageError(f"{path}: expected a mapping at the top level")
  return data


def _require(data: dict, key: str, where: str):
  if key not in data:
    raise GraspStageError(f"{where}: missing '{key}'")
  return data[key]


# -- grasp sets ----------------------------------------------------------------
def _validate_arrays(arrays: dict, ids, spec: HandSpec, where: str) -> GraspSet:
  """Shape / value checks on all grasps at once; errors name the first bad one."""
  g, f = len(arrays["object_pos"]), spec.fingers
  if ids is None:
    ids = np.arange(g, dtype=np.int64)
  ids = np.asarray(ids)
  if ids.shape != (g,) or not np.issubdtype(ids.dtype, np.integer):
    raise GraspStageError(f"{where}: ids must be {g} integers")
  ids = ids.astype(np.int64)
  if g == 0:
    raise GraspStageError(f"{where}: no grasps")
  uniq, counts = np.unique(ids, return_counts=True)
  if (counts > 1).any():
    raise GraspStageError(f"{where}: duplicate id {int(uniq[counts > 1][0])}")
  shapes = {
    "object_pos": (g, 3),
    "object_quat": (g, 4),
    "pose_weights": (g, 2),
    "tip_points": (g, f, 3),
    "tip_normals": (g, f, 3),
    "tip_weights": (g, f),
    "joint_q": (g, spec.finger_joints),
    "joint_weights": (g, f),
  }
  out = {}
  for name, shape in shapes.items():
    try:
      arr = np.asarray(arrays[name], dtype=np.float32)
    except (KeyError, ValueError, TypeError) as e:
      raise GraspStageError(f"{where}: {name}: {e}") from None
    if arr.shape != shape:
      n = int(np.prod(shape[1:]))
      raise GraspStageError(
        f"{where}: {name} needs {n} finite numbers per grasp "
        f"(shape {shape}), got shape {arr.shape}"
      )
    bad = ~np.isfinite(arr).reshape(g, -1).all(-1)
    if bad.any():
      raise GraspStageError(
        f"{where}: {name} of grasp id {ids[bad.argmax()]} not finite"
      )
    out[name] = arr

  def first_bad(mask, name, what):
    if mask.any():
      k = int(np.argmax(mask))
      raise GraspStageError(f"{where}: {name} of grasp id {ids[k]} {what}")

  for name in ("object_quat", "tip_normals"):
    norm = np.linalg.norm(out[name], axis=-1)
    off = np.abs(norm - 1.0) > 1e-3
    first_bad(off.reshape(g, -1).any(-1), name, "is not unit length")
    out[name] = out[name] / norm[..., None]
  for name in ("pose_weights", "tip_weights", "joint_weights"):
    w = out[name]
    first_bad(((w < 0) | (w > 1)).any(-1), name, "must lie in [0, 1]")
  return GraspSet(ids, *(out[a] for a in GraspSet.ARRAYS))


def _check_header(get, hand: str, where: str) -> None:
  if get("version") != SCHEMA_VERSION:
    raise GraspStageError(f"{where}: version must be {SCHEMA_VERSION}")
  if get("hand") != hand:
    raise GraspStageError(f"{where}: hand {get('hand')!r}, training uses {hand!r}")
  if get("units") != _UNITS:
    raise GraspStageError(f"{where}: units must be {_UNITS}")
  if get("frames") != _FRAMES:
    raise GraspStageError(f"{where}: frames must be {_FRAMES}")


def _load_yaml_set(path: Path, hand: str) -> GraspSet:
  data = _read_yaml(path)
  where = str(path)
  _check_header(data.get, hand, where)
  raw = _require(data, "grasps", where)
  if not isinstance(raw, list) or not raw:
    raise GraspStageError(f"{where}: 'grasps' must be a non-empty list")
  cols = {a: [] for a in GraspSet.ARRAYS}
  ids = []
  for k, g in enumerate(raw):
    w = f"{where} grasps[{k}]"
    gid = g.get("id", k)
    if not isinstance(gid, int) or isinstance(gid, bool):
      raise GraspStageError(f"{w}: id must be an integer, got {gid!r}")
    ids.append(gid)
    pose, tip, joint = (_require(g, key, w) for key in ("object_pose", "tip", "joint"))
    pw = pose.get("weights", {"pos": 1.0, "rot": 1.0})
    if not isinstance(pw, dict) or set(pw) != {"pos", "rot"}:
      raise GraspStageError(f"{w}.object_pose.weights: needs exactly pos and rot")
    cols["object_pos"].append(_require(pose, "pos", w + ".object_pose"))
    cols["object_quat"].append(_require(pose, "quat_wxyz", w + ".object_pose"))
    cols["pose_weights"].append([pw["pos"], pw["rot"]])
    cols["tip_points"].append(_require(tip, "points", w + ".tip"))
    cols["tip_normals"].append(_require(tip, "normals", w + ".tip"))
    cols["tip_weights"].append(_require(tip, "weights", w + ".tip"))
    cols["joint_q"].append(_require(joint, "q", w + ".joint"))
    cols["joint_weights"].append(_require(joint, "weights", w + ".joint"))
  return _validate_arrays(cols, ids, HANDS[hand], where)


def _load_npz_set(path: Path, hand: str) -> GraspSet:
  where = str(path)
  with np.load(path, allow_pickle=False) as z:
    data = {k: z[k] for k in z.files}

  def get(key):
    if key == "version":
      return int(data["version"]) if "version" in data else None
    if key in ("units", "frames"):
      prefix = key.rstrip("s") + "_"
      keys = _UNITS if key == "units" else _FRAMES
      return {k: str(data.get(prefix + k, "")) for k in keys}
    return str(data[key]) if key in data else None

  _check_header(get, hand, where)
  if "object_pos" not in data:
    raise GraspStageError(f"{where}: missing 'object_pos'")
  arrays = dict(data)
  arrays.setdefault("pose_weights", np.ones((len(data["object_pos"]), 2), np.float32))
  return _validate_arrays(arrays, data.get("ids"), HANDS[hand], where)


def grasp_set_file(folder: str | Path) -> Path:
  """The ``grasps.npz`` or ``grasps.yaml`` of a grasp-set folder."""
  folder = Path(folder)
  yml, npz = folder / "grasps.yaml", folder / "grasps.npz"
  if yml.is_file() and npz.is_file():
    raise GraspStageError(f"{folder}: has both grasps.yaml and grasps.npz")
  if npz.is_file():
    return npz
  if yml.is_file():
    return yml
  raise GraspStageError(f"{folder}: grasp set not found (grasps.yaml or grasps.npz)")


# Grasp sets read in this process, keyed by file, its mtime / size and hand:
# load_training validates them and the env terms reuse the arrays.
_SET_CACHE: dict[tuple, GraspSet] = {}


def load_grasp_set(folder: str | Path, hand: str) -> GraspSet:
  path = grasp_set_file(folder).resolve()
  stat = path.stat()
  key = (path, stat.st_mtime_ns, stat.st_size, hand)
  if key not in _SET_CACHE:
    load = _load_npz_set if path.suffix == ".npz" else _load_yaml_set
    _SET_CACHE[key] = load(path, hand)
  return _SET_CACHE[key]


def load_grasp_sets(folders, hand: str) -> GraspSet:
  """Several grasp-set folders concatenated (a grasp is identified by its
  folder and id)."""
  sets = [load_grasp_set(f, hand) for f in folders]
  return sets[0] if len(sets) == 1 else GraspSet.concat(sets)


def save_grasp_set_npz(folder: str | Path, grasps: GraspSet, hand: str) -> Path:
  """Write ``grasps`` as ``<folder>/grasps.npz`` (validated on load)."""
  folder = Path(folder)
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / "grasps.npz"
  np.savez(
    path,
    version=np.int64(SCHEMA_VERSION),
    hand=np.str_(hand),
    unit_length=np.str_(_UNITS["length"]),
    unit_angle=np.str_(_UNITS["angle"]),
    frame_object_pose=np.str_(_FRAMES["object_pose"]),
    frame_tip=np.str_(_FRAMES["tip"]),
    ids=np.asarray(grasps.ids, dtype=np.int64),
    **{a: np.asarray(getattr(grasps, a), np.float32) for a in GraspSet.ARRAYS},
  )
  return path


# -- objects and trainings -------------------------------------------------------
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
    encoding = tuple(float(x) for x in enc)
  elif encoding_dim > 0:
    raise GraspStageError(f"{where}: training needs a {encoding_dim}-d encoding")
  return ObjectSpec(name, kind, folder, compound, visual, collision, encoding)


def load_training(path: str | Path) -> TrainingSpec:
  """Load and validate a training file and everything it references. Grasp
  sets are read once per process (``load_grasp_set``)."""
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
    sets, folders = {}, {}
    for key in ("init_grasps", "goal_grasps"):
      per_object = _require(s, key, w)
      if set(per_object) != set(obj_names):
        raise GraspStageError(
          f"{w}.{key}: needs exactly the stage objects {sorted(obj_names)}, "
          f"got {sorted(per_object)}"
        )
      merged, folders[key] = {}, {}
      for n, rels in per_object.items():
        if not rels:
          raise GraspStageError(f"{w}.{key}.{n}: list at least one grasp set")
        paths = tuple((objects[n].folder / rel).resolve() for rel in rels)
        merged[n] = load_grasp_sets(paths, hand)
        folders[key][n] = paths
        role = "init" if key == "init_grasps" else "goal"
        _check_joint_limits(merged[n], hand, role, f"{w}.{key}.{n}")
      sets[key] = merged
    advance = _require(s, "advance", w)
    iters = advance.get("after_iterations") if isinstance(advance, dict) else None
    if not isinstance(iters, int) or iters <= 0:
      raise GraspStageError(f"{w}.advance: after_iterations must be a positive int")
    stages.append(
      StageSpec(
        name,
        stage_objects,
        sets["init_grasps"],
        sets["goal_grasps"],
        iters,
        folders["init_grasps"],
        folders["goal_grasps"],
      )
    )
  return TrainingSpec(path, hand, enc_name, enc_dim, tuple(stages))
