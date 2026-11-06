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

## `make_pose_grasps.py`

Temporary test helper (not part of the training pipeline): random object-pose grasp sets that imitate the reorient pose-goal task with the grasp-stage pipeline. Only the object pose varies; tip and joint weights are 0.

```bash
pixi run python src/tools/make_pose_grasps.py \
  --object-dir src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube \
  --name cube_6d_10k --count 10000 [--seed 0] [--lift 0.0] [--pos-noise 0.01] \
  [--stages 5] [--max-angles 36,72,108,144,180] [--training] [--iterations N]
```

- `init`: the reorient cube's spawn point (+ `--lift`) ± `--pos-noise` per world axis, random orientation, finger joints from the reorient reset ranges.
- `goal`: position uniform in the reorient goal box (wrist-tag frame), random orientation, pose weights 1.
- Orientations (init and goal alike) are uniform within a maximum angle of the object's default start orientation (its spawn orientation at a reorient reset), so a grasp pair asks for a rotation of at most twice that angle; 180° is uniform on SO(3).
- Single set (default): uniform on SO(3), writes `<object-dir>/grasps/<name>/{init,goal}/grasps.npz`; `--training` writes `trainings/<name>.yaml` (one stage, 5000 iterations by default).
- `--stages K`: a range curriculum `<name>_stage1` … `<name>_stageK`, maximum angle growing evenly to 180° (K = 5: 36°, 72°, 108°, 144°, 180°) or as given by `--max-angles`; `--training` writes `trainings/<name>_curriculum.yaml` (K stages, 1000 iterations each by default).
- `trainings/` is created next to the objects folder (`<object-dir>/../../trainings`); every written set and training file is validated.
- Example in `src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/` (see its `README.md`; the npz sets are not in git, generate them with the commands there): sets `cube_6d_10k`, `cube_6d_1k`, `cube_6d_10k_stage1..5` and trainings `cube_6d_10k.yaml`, `cube_6d_1k.yaml`, `cube_6d_10k_curriculum.yaml`, e.g.
  `pixi run train-stages --training src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/trainings/cube_6d_10k_curriculum.yaml`.
