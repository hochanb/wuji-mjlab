# Cube 6D pose example

The reorient cube pose-goal task (rotate and move the cube to a goal pose)
expressed as grasp-stage files, for testing `WujiHand2_GraspStage_50Hz`. There
is no hand-shape target: only the object pose is set, and tip / joint weights
are 0. File formats are described in
[`docs/grasp_stage.md`](../../../../tasks/reorient/docs/grasp_stage.md).

## Contents

```
cube_6d/
  trainings/
    cube_6d_1k.yaml               # 1 stage, 1k grasps, 5000 iterations
    cube_6d_10k.yaml              # 1 stage, 10k grasps, 5000 iterations
    cube_6d_10k_curriculum.yaml   # 5 stages x 1000 iterations (rotation range)
  objects/cube/
    object.yaml                   # 54 mm / 120 g box (compound "cube")
    grasps/<set>/{init,goal}/grasps.npz   # generated, not in git
```

The npz sets are not in git (`.gitignore`): generate them first (below).

| Set | Grasps | Max angle from the start orientation |
|---|---|---|
| `cube_6d_1k`, `cube_6d_10k` | 1k, 10k | 180° (uniform on SO(3)) |
| `cube_6d_10k_stage1` … `_stage5` | 10k each | 36°, 72°, 108°, 144°, 180° |

- **init**: the cube at the reorient spawn point ±1 cm per world axis; finger
  joints from the reorient reset ranges.
- **goal**: position uniform in the reorient goal box (wrist-tag frame); pose
  weights 1, so position and orientation are both required.
- Orientations of init and goal are uniform within the set's maximum angle of
  the cube's default start orientation, so one episode asks for a rotation of
  at most twice that angle. Stage 5 equals the plain sets.

## Generating the sets

From the repo root, with `src/tools/make_pose_grasps.py`. With the default
`--seed 0`, these commands always produce the same files. Each command
overwrites its sets and training file and validates them. A run takes about
half a minute.

```bash
OBJ=src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube

# single sets (uniform orientation) + trainings/cube_6d_{10k,1k}.yaml
pixi run python src/tools/make_pose_grasps.py --object-dir $OBJ \
  --name cube_6d_10k --count 10000 --training
pixi run python src/tools/make_pose_grasps.py --object-dir $OBJ \
  --name cube_6d_1k --count 1000 --training

# 5-stage curriculum: cube_6d_10k_stage1..5 + trainings/cube_6d_10k_curriculum.yaml
pixi run python src/tools/make_pose_grasps.py --object-dir $OBJ \
  --name cube_6d_10k --count 10000 --stages 5 --training
```

Options:

- `--max-angles 30,60,90,135,180`: per-stage angles in degrees, one per stage.
- `--iterations N`: iterations per stage in the training file. The default is
  1000 for a curriculum and 5000 for a single set.
- `--seed`: random seed. Stage k uses `seed + k - 1`.
- `--pos-noise`: spawn noise in metres (default 0.01).
- `--lift`: spawn height offset in metres.
- A new `--name` writes new sets next to these. Do not overwrite a set that a
  past run used: the run records the set's SHA-256 in `params/grasp_sets.yaml`.

## Training

```bash
T=src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/trainings

# curriculum: 5 stages x 1000 iterations
CUDA_VISIBLE_DEVICES=0 pixi run train-stages --training $T/cube_6d_10k_curriculum.yaml

# single stage, uniform orientation from the start
CUDA_VISIBLE_DEVICES=0 pixi run train-stages --training $T/cube_6d_10k.yaml
```

- Overrides go after the training file, e.g. `env.scene.num_envs=4096`,
  `agent.run_name=cube_curriculum` or `agent.seed=1`.
- The total number of iterations is the sum of the stages, not
  `agent.max_iterations`.
- Logs go to `logs/rsl_rl/wuji_grasp_stage_hand2/<date>_<run_name>/`, with a
  checkpoint `model_<it>_stage<i>_<name>.pt` at each stage switch.
- To change the stage lengths, edit `advance.after_iterations` in the training
  file, or regenerate the file with `--iterations`.

Differences from the original reorient cube task: there is no cube-size
domain randomisation, and the rotation range is set by the stages instead of
the `goal_rotation` curriculum.
