"""Put an arbitrary object on the Wuji Hand 2 in place of the reorient cube.

The object is either a mesh file (``.obj``, or a ``.urdf`` whose first mesh and
mass are used) or a named primitive object: ``box`` / ``hammer`` defined here,
or a repo compound object such as ``stapler``. Mesh objects get their pivot
(body origin) moved to the mesh centre of mass via a COM-centred copy of the
OBJ, and a ``map_Kd`` texture from the OBJ's ``.mtl`` is applied if present.
MuJoCo collides meshes as their convex hull.

Usage (from the repo root):
  pixi run python src/tools/hand_object_scene.py check <object> [<object> ...]
  pixi run python src/tools/hand_object_scene.py view <object> [--orient lying_x] [--port 8082]

``check`` drops the object onto the idle hand in each candidate orientation and
reports whether it stays (3 s); ``view`` opens the scene in a viser viewer bound
to 127.0.0.1. The object reuses the cube's body/geom/site names (cube,
cube_visual, cube_center) so the reorient task terms keep working unchanged.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.play import PlayConfig
from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  COMPOUND_OBJECTS,
)
from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  compound_spec as repo_compound_spec,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_CUBE_INIT_POS,
)
from wuji_mjlab.utils.play_runner import run_play_with_cfg
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

REPO = Path(__file__).resolve().parents[2]
MESH_CACHE = REPO / "tmp" / "mesh_cache"  # gitignored
TASK = "WujiHand2_Reorient_50Hz"
CUBE_INIT_POS = np.array(REORIENT_HAND2_CUBE_INIT_POS["right"])
DROP_HEIGHT = 0.04  # object pivot starts this far above the cube's start, then settles
DEFAULT_MESH_RGBA = (0.9, 0.62, 0.15, 1.0)
CONTACT = dict(  # contact parameters of the reorient cube (xmls/cube.xml)
  friction=[0.3, 0.005, 0.0001],
  conaffinity=2,
  condim=3,
  solref=[0.015, 1.0],
  solimp=[0.85, 0.85, 0.001, 0.5, 2.0],
)


# --- Mesh objects -------------------------------------------------------------


@dataclass(frozen=True)
class MeshSource:
  obj: Path
  mass: float
  texture: Path | None


def _urdf_mesh_and_mass(urdf: Path) -> tuple[Path, float | None]:
  root = ET.parse(urdf).getroot()
  mesh = root.find(".//visual/geometry/mesh")
  if mesh is None:
    mesh = root.find(".//mesh")
  if mesh is None:
    raise ValueError(f"{urdf}: no <mesh> element")
  mass = root.find(".//inertial/mass")
  return (urdf.parent / mesh.get("filename")).resolve(), (
    float(mass.get("value")) if mass is not None else None
  )


def _obj_texture(obj: Path) -> Path | None:
  """map_Kd image of the OBJ's first .mtl, if any."""
  for line in obj.read_text(errors="ignore").splitlines():
    if line.startswith("mtllib"):
      mtl = obj.parent / line.split(maxsplit=1)[1].strip()
      if mtl.exists():
        m = re.search(r"^\s*map_Kd\s+(.+?)\s*$", mtl.read_text(errors="ignore"), re.M)
        if m and (obj.parent / m.group(1)).exists():
          return obj.parent / m.group(1)
      break
  return None


def mesh_source(path: Path, mass: float | None) -> MeshSource:
  if path.suffix.lower() == ".urdf":
    obj, urdf_mass = _urdf_mesh_and_mass(path)
    mass = mass if mass is not None else urdf_mass
  else:
    obj = path.resolve()
  if mass is None:
    raise ValueError(f"{path}: no mass given (--mass) and none in the file")
  return MeshSource(obj=obj, mass=mass, texture=_obj_texture(obj))


def _mesh_com(path: Path) -> np.ndarray:
  """Mesh centre of mass in the OBJ frame (MuJoCo's own computation)."""
  spec = mujoco.MjSpec()
  spec.add_mesh(name="m", file=str(path))
  body = spec.worldbody.add_body(name="b")
  body.add_freejoint()
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m", mass=1.0)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return data.xipos[1].copy()  # body at origin with identity rotation


def centered_obj(src: Path) -> Path:
  """COM-centred copy of an OBJ (cached under tmp/mesh_cache)."""
  key = f"{src.stem}_{abs(hash((str(src), src.stat().st_mtime_ns))) % 10**8}"
  dst = MESH_CACHE / f"{key}_centered.obj"
  if not dst.exists():
    MESH_CACHE.mkdir(parents=True, exist_ok=True)
    com = _mesh_com(src)
    lines = []
    for line in src.read_text(errors="ignore").splitlines():
      if line.startswith("v "):
        x, y, z = (float(t) for t in line.split()[1:4])
        line = f"v {x - com[0]:.6f} {y - com[1]:.6f} {z - com[2]:.6f}"
      elif line.startswith("mtllib"):
        continue  # MuJoCo ignores materials; the texture is attached explicitly
      lines.append(line)
    dst.write_text("\n".join(lines) + "\n")
  return dst


def mesh_object_spec(src: MeshSource, rgba=DEFAULT_MESH_RGBA) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.add_mesh(name="object_mesh", file=str(centered_obj(src.obj)))
  material = ""
  if src.texture is not None:
    spec.add_texture(
      name="object_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(src.texture)
    )
    mat = spec.add_material(name="object_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "object_tex"
    material, rgba = "object_mat", (1.0, 1.0, 1.0, 1.0)
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_freejoint")
  body.add_geom(
    name="cube_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="object_mesh",
    contype=0,
    conaffinity=0,
    density=0,
    group=2,
    rgba=list(rgba),
    material=material,
  )
  body.add_geom(
    name="cube",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="object_mesh",
    mass=src.mass,
    group=3,
    **CONTACT,
  )
  body.add_site(name="cube_center", pos=[0.0, 0.0, 0.0], group=4)
  return spec


# --- Primitive objects ----------------------------------------------------------

# Boxes from the reorient cube.xml (texture + contact params kept):
# name -> per-axis scale of the 54 mm cube.
PRIMITIVE_BOXES = {"box": (1.2, 1.5, 0.8)}

# Primitive compounds, in the object frame [m] before COM centring.
# Each part: (kind, size, pos, density, rgba); box size = half extents,
# cylinder size = (radius, half length) along the part's x axis.
_CUBE_HALF = 0.027
COMPOUNDS = {
  "hammer": [
    # head: 0.72 x 1.8 x 0.72 of the 54 mm cube (long axis y)
    (
      "box",
      (0.72 * _CUBE_HALF, 1.8 * _CUBE_HALF, 0.72 * _CUBE_HALF),
      (0.0, 0.0, 0.0),
      762.0,
      (0.3, 0.3, 0.32, 1.0),
    ),
    # handle: 20.48 mm x 144 mm cylinder along +x, starting at the head centre
    ("cylinder", (0.01024, 0.072), (0.072, 0.0, 0.0), 600.0, (0.72, 0.52, 0.3, 1.0)),
  ],
}
_CYL_ALONG_X = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]  # z-axis cylinder -> x


def box_spec(scale) -> mujoco.MjSpec:
  from wuji_mjlab.assets.objects.inhand_object.object_cfg import (
    BASELINE_EDGE_M,
    CUBE_DENSITY,
    INHAND_OBJECT_XML,
  )

  spec = mujoco.MjSpec.from_file(str(INHAND_OBJECT_XML))
  half = [BASELINE_EDGE_M / 2 * k for k in scale]
  for geom in spec.geoms:
    if geom.name in ("cube", "cube_visual"):
      geom.size = half
  next(g for g in spec.geoms if g.name == "cube").mass = (
    CUBE_DENSITY * 8 * half[0] * half[1] * half[2]
  )
  return spec


def compound_spec(name: str) -> mujoco.MjSpec:
  """Primitive parts as separate geoms; the body origin (pivot) is the total COM."""
  parts = COMPOUNDS[name]
  masses = np.array(
    [
      d * (8 * np.prod(s) if k == "box" else np.pi * s[0] ** 2 * 2 * s[1])
      for k, s, _, d, _ in parts
    ]
  )
  com = (masses[:, None] * np.array([p[2] for p in parts])).sum(0) / masses.sum()
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_freejoint")
  for i, ((kind, size, pos, _, rgba), mass) in enumerate(
    zip(parts, masses, strict=True)
  ):
    is_box = kind == "box"
    common = dict(
      type=mujoco.mjtGeom.mjGEOM_BOX if is_box else mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=list(size) if is_box else [size[0], size[1], 0.0],
      pos=(np.array(pos) - com).tolist(),
      quat=[1.0, 0.0, 0.0, 0.0] if is_box else _CYL_ALONG_X,
    )
    suffix = "" if i == 0 else f"_{i}"
    body.add_geom(
      name=f"cube{suffix}_visual",
      contype=0,
      conaffinity=0,
      density=0,
      group=2,
      rgba=rgba,
      **common,
    )
    body.add_geom(name=f"cube{suffix}", mass=float(mass), group=3, **CONTACT, **common)
  body.add_site(name="cube_center", pos=[0.0, 0.0, 0.0], group=4)
  return spec


def object_spec(name: str, mass: float | None = None) -> mujoco.MjSpec:
  if name in COMPOUND_OBJECTS:
    return repo_compound_spec(name)
  if name in PRIMITIVE_BOXES:
    return box_spec(PRIMITIVE_BOXES[name])
  if name in COMPOUNDS:
    return compound_spec(name)
  path = Path(name)
  if path.suffix.lower() in (".obj", ".urdf") and path.exists():
    return mesh_object_spec(mesh_source(path, mass))
  known = sorted({*COMPOUND_OBJECTS, *PRIMITIVE_BOXES, *COMPOUNDS})
  raise ValueError(
    f"unknown object {name!r}: give an .obj/.urdf path or one of {known}"
  )


# --- Scene ----------------------------------------------------------------------


def _rot(axis, deg) -> np.ndarray:
  half = np.radians(deg) / 2
  return np.array([np.cos(half), *(np.sin(half) * np.asarray(axis, dtype=float))])


# Candidate resting orientations (world frame, wxyz) of the object frame.
ORIENTATIONS = {
  "upright": np.array([1.0, 0.0, 0.0, 0.0]),
  "lying_x": _rot((1, 0, 0), 90),
  "lying_y": _rot((0, 1, 0), 90),
}


def make_env_cfg(name: str, orient: str, mass: float | None = None, num_envs: int = 1):
  env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
  object_spec(name, mass)  # fail early on a bad object; EntityCfg rebuilds it
  env_cfg.scene.entities["object"] = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=tuple((CUBE_INIT_POS + [0.0, 0.0, DROP_HEIGHT]).tolist()),
      rot=tuple(ORIENTATIONS[orient].tolist()),
    ),
    spec_fn=lambda: object_spec(name, mass),
  )
  # Fixed object pose: reset to the default root state with no random object
  # orientation (the reset term is what writes the pose). The hand keeps the
  # task's fixed mount tilt.
  env_cfg.events["reset_object_pose"] = EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={"pose_range": {}, "asset_cfg": SceneEntityCfg("object")},
  )
  env_cfg.scene.num_envs = num_envs
  return env_cfg, agent_cfg


def describe(name: str, mass: float | None = None) -> None:
  model = object_spec(name, mass).compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  b = model.body("cube").id
  print(
    f"{name}: mass = {model.body_mass[b]:.3f} kg, COM offset from pivot [mm] = "
    f"{np.round(1000 * model.body_ipos[b], 3)}, ngeom = {model.ngeom}"
  )


def check(names: list[str], mass: float | None = None) -> None:
  for name in names:
    describe(name, mass)
    for orient in ORIENTATIONS:
      env_cfg, agent_cfg = make_env_cfg(name, orient, mass)
      env_cfg.terminations = {}
      env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
      env.max_common_steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env
      env.reset()
      obj, robot = env.scene["object"], env.scene["robot"]
      palm_id = robot.find_bodies(".*_palm_link")[0][0]
      zero = torch.zeros(1, env.action_manager.total_action_dim, device="cuda:0")
      for _ in range(150):  # 3 s at 50 Hz with the hand holding its default pose
        env.step(zero)
      com = obj.data.root_link_pos_w[0].cpu().numpy()  # pivot == COM
      d = com - robot.data.body_link_pos_w[0, palm_id].cpu().numpy()
      speed = float(obj.data.root_link_lin_vel_w[0].norm())
      on_hand = d[2] > -0.02 and np.linalg.norm(d[:2]) < 0.12 and speed < 0.05
      print(
        f"  {orient:8s} COM - palm [cm] = {np.round(100 * d, 1)}, speed = {speed:.3f} m/s "
        f"-> {'ON HAND' if on_hand else 'fell/moving'}"
      )
      env.close()


def _object_frame_only(self, *, visualizer, num_envs, object_pos_w, object_quat_w, **_):
  """Replaces the reorient goal drawing: this scene has no goal, so only the
  object's own frame (at its COM pivot) is drawn."""
  from mjlab.utils.lab_api.math import matrix_from_quat

  rot = matrix_from_quat(object_quat_w).detach().cpu().numpy()
  for env_idx in visualizer.get_env_indices(num_envs):
    visualizer.add_frame(
      position=object_pos_w[env_idx].detach().cpu().numpy(),
      rotation_matrix=rot[env_idx],
      scale=0.06,
      axis_radius=0.003,
      alpha=1.0,
      label=f"object_frame_env{env_idx}",
    )


def view(name: str, orient: str, port: int, mass: float | None = None) -> None:
  import viser
  from mjlab.viewer import ViserPlayViewer
  from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
    ReorientCommandVisualization,
  )

  ReorientCommandVisualization._draw_primitive_goal = _object_frame_only
  env_cfg, agent_cfg = make_env_cfg(name, orient, mass)
  original_init = ViserPlayViewer.__init__

  def init_local(self, *args, **kwargs):
    kwargs.setdefault("viser_server", viser.ViserServer(host="127.0.0.1", port=port))
    original_init(self, *args, **kwargs)

  ViserPlayViewer.__init__ = init_local
  cfg = PlayConfig(agent="zero", viewer="viser", no_terminations=True)
  run_play_with_cfg(TASK, cfg, env_cfg, agent_cfg)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = parser.add_subparsers(dest="cmd", required=True)
  p_check = sub.add_parser("check", help="headless: does it stay on the idle hand?")
  p_check.add_argument("objects", nargs="+", help=".obj/.urdf path or object name")
  p_view = sub.add_parser("view", help="viser viewer of the hand + object")
  p_view.add_argument("object", help=".obj/.urdf path or object name")
  p_view.add_argument("--orient", choices=sorted(ORIENTATIONS), default="upright")
  p_view.add_argument("--port", type=int, default=8082)
  for p in (p_check, p_view):
    p.add_argument("--mass", type=float, default=None, help="mesh mass [kg]")
  args = parser.parse_args()
  if args.cmd == "check":
    check(args.objects, args.mass)
  else:
    view(args.object, args.orient, args.port, args.mass)


if __name__ == "__main__":
  main()
