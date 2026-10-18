"""Show .obj / .urdf meshes side by side in a viser viewer (127.0.0.1 only).

URDFs are compiled with MuJoCo first as a load check, then their first mesh is
shown. OBJ textures (``map_Kd`` in the ``.mtl``) are kept via trimesh.

Usage (from the repo root):
  pixi run python src/tools/view_mesh.py <mesh.obj|robot.urdf> [...] [--port 8082]
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

_PALETTE = [(230, 160, 40), (90, 140, 220), (110, 190, 110), (200, 90, 160)]


def _mesh_path(path: Path) -> Path:
  if path.suffix.lower() != ".urdf":
    return path
  model = mujoco.MjModel.from_xml_path(str(path))
  print(f"[{path.name}] mujoco load ok: nbody={model.nbody} nmesh={model.nmesh}")
  mesh = ET.parse(path).getroot().find(".//mesh")
  if mesh is None:
    raise ValueError(f"{path}: no <mesh> element")
  return path.parent / mesh.get("filename")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("paths", nargs="+", type=Path)
  parser.add_argument("--port", type=int, default=8082)
  parser.add_argument("--spacing", type=float, default=0.25, help="x spacing [m]")
  args = parser.parse_args()

  server = viser.ViserServer(host="127.0.0.1", port=args.port)
  server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05)
  server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002)
  x0 = -args.spacing * (len(args.paths) - 1) / 2
  for i, path in enumerate(args.paths):
    mesh = trimesh.load(_mesh_path(path), force="mesh")
    print(
      f"[{path.name}] {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
      f"extent [cm] = {np.round(100 * mesh.extents, 1)}, watertight = {mesh.is_watertight}"
    )
    if not isinstance(mesh.visual, trimesh.visual.TextureVisuals):
      mesh.visual = trimesh.visual.ColorVisuals(
        mesh, face_colors=np.array([*_PALETTE[i % len(_PALETTE)], 255])
      )
    x = x0 + i * args.spacing
    server.scene.add_mesh_trimesh(f"/mesh_{i}", mesh, position=(x, 0.0, 0.0))
    server.scene.add_label(
      f"/label_{i}", path.name, position=(x, 0.0, float(mesh.bounds[1][2]) + 0.03)
    )
  print(f"viser: http://localhost:{args.port}")
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
