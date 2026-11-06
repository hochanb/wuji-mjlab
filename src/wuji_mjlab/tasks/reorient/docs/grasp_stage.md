# Grasp Stages · Config Files

[Task home](../README.md) · [Train](train.md)

How to write and edit the files behind `WujiHand2_GraspStage_50Hz`: training
files (stages), objects and grasp sets. The loader and every rule below live in
`src/wuji_mjlab/tasks/reorient/grasp_stage/schema.py`; the env built from a
stage is in `config/wuji_hand2/grasp_stage.py`.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 pixi run train-stages --training <training.yaml> \
  [env.<path>=<value> ...] [agent.<path>=<value> ...]
```

Every stage is validated before the first env is built; a bad file stops the
run with the file path, the field and, for grasp arrays, the first bad grasp
id. Each stage gets a fresh env; the policy, optimiser and iteration count
carry over and a checkpoint `model_<it>_stage<i>_<name>.pt` is written at each
stage switch. The run folder's `params/` keeps `training.yaml`, the object
files, `grasp_sets.yaml` (every grasp set used: file path, SHA-256, size and
grasp count; the sets themselves are not copied), `env_stage<i>_<name>.yaml`
(grasp sets appear as folder paths) and `obs_layout_stage<i>_<name>.yaml`.
Do not change or move a grasp set a run used: the checksum identifies it.

The bundled example is `src/wuji_mjlab/assets/grasp_stage/`.

## Layout

```
<root>/
  trainings/
    my_training.yaml          # object_root: ../objects
  objects/
    stapler/                  # folder name == object name
      object.yaml
      encoding.npy            # optional, see "Object"
      grasps/
        init_rest/grasps.yaml # a grasp set: grasps.yaml OR grasps.npz
        goal_side/grasps.npz
```

Paths in a training file are resolved from the training file's folder
(`object_root`, default `objects`); grasp-set paths from the object folder.

## Training file

```yaml
version: 1
hand: wuji_hand2_right            # the only hand for now
encoding: {name: none, dim: 0}    # dim > 0 needs an encoding per object
object_root: ../objects
stages:
  - name: s0                      # unique
    objects: [stapler]            # one object per stage for now
    init_grasps: {stapler: [grasps/init_rest]}
    goal_grasps: {stapler: [grasps/goal_side, grasps/goal_top]}  # merged
    advance: {after_iterations: 1000}
```

- Stages run in order; for now a stage ends after `advance.after_iterations`
  (success-based advancing comes later). The run's total iterations is their
  sum.
- `init_grasps` and `goal_grasps` must list every stage object, each with at
  least one grasp set. Several sets are concatenated; a grasp is identified
  by its set folder and id, so ids may repeat across sets. A set is read once
  per process and handed to the env as arrays (never Python lists).
- Every stage of a training must give the same observation layout: same
  hand, same `encoding.dim`. Grasp scheme parts (tip / joint / pose) are always
  in the observations and switched by their weights, so stages may differ in
  which parts they use.
- Keep the env count the same across stages (`env.scene.num_envs` applies to
  all): the PPO storage is reused.

## Object

```yaml
name: stapler                 # must equal the folder name
kind: compound                # compound | mesh
compound: stapler             # builder in compound_objects.COMPOUND_OBJECTS
# kind: mesh
# mesh: {visual: visual.obj, collision: [col_000.obj, col_001.obj]}
# encoding: encoding.npy      # 1-D float array, length = training encoding.dim
```

- `compound`: built from primitives by a registered builder; add a builder to
  `assets/objects/inhand_object/compound_objects.py` for a new one. The body
  origin is the object's centre of mass; grasp poses and tip targets refer to
  that frame.
- `mesh`: visual mesh plus convex collision parts (a convex decomposition;
  mujoco-warp collides convex hulls only). Validated, but not built yet.
- A stage cannot mix `mesh` and `compound` objects.
- `encoding`: precomputed (not at runtime) by one encoder E for every object
  of the training; required when `encoding.dim > 0`.

## Grasp set

A folder holding exactly one of `grasps.yaml` (hand-edited) or `grasps.npz`
(generated). Both are handled the same way and sized for millions of grasps:
3 million grasps are ~850 MB as npz and load, validate and reach the env in a
few seconds; YAML parses far slower (~6 s per 10k grasps), so generate npz.

### Conventions

| Item | Frame / unit | Notes |
|---|---|---|
| `object_pose.pos` | hand base frame (`right_palm_link`), m | object COM origin |
| `object_pose.quat_wxyz` | hand base frame | unit (tolerance 1e-3, renormalised) |
| `object_pose.weights` | `{pos, rot}` in [0, 1] | optional, default 1 |
| `tip.points` | object frame, m | 5 rows, thumb → pinky, **on the object surface** |
| `tip.normals` | object frame | 5 unit rows, pointing **out** of the surface |
| `tip.weights` | [0, 1] | 5 values |
| `joint.q` | rad | 20 finger joints, `right_finger1_joint1..4` → `right_finger5_joint1..4`; wrist excluded |
| `joint.weights` | [0, 1] | 5 values, one per finger |

- The hand base frame is the palm body, i.e. the wrist in the sense of the
  hand's own base, not the world. The file header says `object_pose: wrist`.
- Tip reference point: each fingertip is represented by a point 5 mm inside
  its pad (`HAND2_TIP_PAD_POINTS`, `pad_depth = 0.005`). When the pad touches
  a target, that point sits at `point + 0.005 * normal`; the tip error is the
  offset from there, so it is zero for a touching pad. Do not offset the
  target points yourself.
- Joint angles are checked against the hand model's limits
  (`schema.hand_joint_limits`): an initial grasp's angles must lie within each
  joint's range (the hand is put there); a goal grasp's angles, for fingers
  with `w_joint > 0`, within the soft limits (90 % of the range, the limits
  the env's joint targets and observations use), so the goal is reachable.

### Weights

A weight of 0 switches that part off; values between 0 and 1 (e.g. 0.3) are
importance.

| Where | Use |
|---|---|
| Success G | only `w > 0` matters: required or not |
| Rewards | multiplied by `w` (orientation / position alignment, weighted tip and joint terms) |
| Observations | parts with `w = 0` read exactly 0; the weights themselves are observed |

Success G: object orientation within 0.2 rad (`w_rot > 0`) and position within
1 cm (`w_pos > 0`), every tip with `w_tip > 0` within 10 mm of its contact
point and within 60° of the normal, every finger with `w_joint > 0` within
0.2 rad on all its joints, held for 13 steps.

### Initial vs goal grasps

- **Initial** grasps place the hand and the object at reset: `joint.q` sets
  the finger joints and `object_pose` the object (the wrist starts at zero).
  They must be full (joint angles always) and free of hand-object overlap;
  the generating module is responsible for that. Their tip / joint weights
  are unused.
- **Goal** grasps: one is drawn per episode (and again after each success
  window). Their weights define what success requires.

### YAML

```yaml
version: 1
hand: wuji_hand2_right
units: {length: m, angle: rad}
frames: {object_pose: wrist, tip: object}
grasps:
  - id: 0                               # optional integer, unique in the set
    object_pose:
      pos: [0.0088, 0.03066, -0.06066]
      quat_wxyz: [0.91398, 0.0018, 0.40309, 0.04649]
      weights: {pos: 1, rot: 1}         # optional
    tip:
      points:  [[...], [...], [...], [...], [...]]
      normals: [[...], [...], [...], [...], [...]]
      weights: [1, 1, 1, 1, 1]
    joint:
      q: [0.0, ...]                     # 20 values
      weights: [0, 0, 0, 0, 0]
```

### npz

Write it with the helper rather than by hand:

```python
from wuji_mjlab.tasks.reorient.grasp_stage.schema import GraspSet, save_grasp_set_npz

grasps = GraspSet(
  ids,            # (G,) integers, unique in the set, e.g. np.arange(G)
  object_pos,     # (G, 3)
  object_quat,    # (G, 4) wxyz
  pose_weights,   # (G, 2) (w_pos, w_rot)
  tip_points,     # (G, 5, 3)
  tip_normals,    # (G, 5, 3)
  tip_weights,    # (G, 5)
  joint_q,        # (G, 20)
  joint_weights,  # (G, 5)
)
save_grasp_set_npz("objects/stapler/grasps/goal_side", grasps, "wuji_hand2_right")
```

Keys in the file: `version`, `hand`, `unit_length`, `unit_angle`,
`frame_object_pose`, `frame_tip`, `ids` and the eight arrays above (`ids` may
be omitted: row index; `pose_weights` too: all 1). Arrays are stored and used
as float32, ids as int64.

## Observations

Each group is ordered `[env obs, condition]`; `obs_layout_*.yaml` lists every
term's slice.

- env obs: measured from the environment, with the policy's sensor noise
  (training only): joints, the object pose errors, the masked tip errors
  (±8 mm), action history, wrist pose; the critic adds privileged state.
- condition: `grasp_weights` (w_pos, w_rot, w_tip ×5, w_joint ×5),
  `grasp_joint_goal` (normalised like the joint observation) and
  `object_encoding`; never noised.

Policy 329 / critic 574 dims with `encoding.dim = 0`.

## Checklist and common errors

- [ ] Object folder name equals `name` in `object.yaml`.
- [ ] Each grasp-set folder has `grasps.yaml` or `grasps.npz`, not both.
- [ ] `version: 1`, `hand`, `units`, `frames` exactly as above.
- [ ] 5 tip rows (thumb first), 20 joint angles, 5 weights per part.
- [ ] Unit quaternions and normals; normals point out of the surface.
- [ ] Tip points on the surface (not offset by the pad depth).
- [ ] Integer ids, unique within each set (or omitted).
- [ ] Initial grasps have real joint angles (within the joint ranges) and no
      hand-object overlap.
- [ ] Goal joint angles of weighted fingers within the soft limits.
- [ ] Same `encoding.dim` and hand for every stage; encodings exist when
      `dim > 0`.

| Error | Cause |
|---|---|
| `... needs 20 finite numbers per grasp` | wrong array length, or NaN / inf |
| `... is not unit length` | quaternion or normal not normalised (beyond 1e-3) |
| `... must lie in [0, 1]` | a weight out of range |
| `grasp set not found` / `file not found` | path relative to the wrong folder |
| `has both grasps.yaml and grasps.npz` | keep one |
| `duplicate id N` | the same id twice in one set |
| `id must be an integer` | a non-integer id |
| `... is outside its range [...]` | initial grasp joint beyond the joint range |
| `... is outside its soft limits [...]` | weighted goal joint beyond the soft limits |
| `stages with several objects are not supported yet` | one object per stage for now |

## Not supported yet

Several objects per stage, mesh objects, success-based stage advancing,
generated goals (random orientation / position boxes) instead of grasp sets.
