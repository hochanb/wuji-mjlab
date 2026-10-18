# Tools

Object/mesh helpers for the reorient task. Run from the repo root. Viewers bind to `127.0.0.1`; forward the port over SSH (`ssh -L 8082:localhost:8082 ...`).

Scratch data (downloaded meshes, caches) goes in the gitignored `tmp/`.

## `hand_object_scene.py`

Puts an object on the Wuji Hand 2 in place of the reorient cube (hand idle, no policy).

```bash
pixi run python src/tools/hand_object_scene.py check <object> [<object> ...]
pixi run python src/tools/hand_object_scene.py view <object> [--orient upright|lying_x|lying_y] [--port 8082]
```

- `<object>`: an `.obj` / `.urdf` path, or a name: `box`, `hammer` (defined in the script), `stapler` (repo asset).
- URDF: uses its first mesh and its mass. OBJ: pass `--mass <kg>`. A `map_Kd` texture in the `.mtl` is applied.
- Mesh pivot is moved to the centre of mass (centred copy cached in `tmp/mesh_cache/`). Collision is the mesh's convex hull.
- `check`: drops the object in each orientation and reports whether it stays on the hand after 3 s, plus mass and pivot/COM offset.
- `view`: viser viewer; draws only the object's frame (no goal).

## `view_mesh.py`

Shows `.obj` / `.urdf` meshes side by side (URDFs are load-checked with MuJoCo first).

```bash
pixi run python src/tools/view_mesh.py <mesh.obj|robot.urdf> [...] [--port 8082]
```
