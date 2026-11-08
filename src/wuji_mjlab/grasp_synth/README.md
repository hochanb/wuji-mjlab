# Grasp synthesis (Wuji Hand 2)

Two tools after Dexonomy ([arXiv:2504.18829](https://arxiv.org/abs/2504.18829)).
Every attempt is one mujoco-warp world, so attempts run in parallel on one GPU.

- `DexonomySynthesizer`: the object pose is an output. A hand shape
  (template) is fitted around the object, then refined and validated.
- `GraspSynthesizer`: the object pose in the wrist frame is an input; only
  the fingers are searched.

## DexonomySynthesizer (object pose as output)

```python
from wuji_mjlab.grasp_synth import DexonomyCfg, DexonomySynthesizer, GraspObject

dex = DexonomySynthesizer(DexonomyCfg(
  wrist_quat_w=(0.541675, 0.45452, 0.45452, 0.541675),  # default: training env wrist
))
obj = GraspObject.from_file("mug.obj", mass=0.15)
res = dex.generate(obj, "tripod", k=8)          # library templates of the type
res = dex.generate(obj, "tripod", k=8, hand_q=q20)  # or a given hand shape (pads)
res.object_pose   # (n <= 8, 7) object pose in the wrist frame, pos + wxyz
res.joint_pos     # (n, 20) finger joints; res.joint_target: squeeze targets
many = dex.generate_many(obj, ["pinch", "tripod", "power_palm"], k=16)  # {type: result}
```

`generate_many` runs several library grasp types of one object in the same
alignment and verification batches. Each type keeps its own budget and
results. It takes about as long as the slowest type alone. A type stops
after `stall_rounds` rounds without a candidate, i.e. when its templates do
not fit the object. Verification runs on a simulation sized to the batch
(64 / 256 / 1024 worlds), since a step on thousands of worlds costs far more.

- **Input:** the wrist orientation in the world. Gravity is always world -z,
  so the orientation only sets the gravity direction relative to the hand.
  The object is fixed at the identity.
- **Output:** the result expressed in the wrist frame, with the wrist at that
  orientation.
- **Templates** (`templates.yaml`, `templates.py`): a template is a hand shape
  plus the link surface points and normals that touch. The default library
  was made from validated cube grasps with `template_from_grasp`.
- **Template update:** with `grow_templates=True`, every validated grasp
  becomes a template of its type, after Dexonomy's template update. It must
  touch with the source template's fingers and differ by 0.1 rad on some
  joint; there are at most `max_templates` per type. New templates are used
  from the next round on and stay in `dex.library` for later calls (other
  objects). Nothing is written until `dex.save_templates(path)`; the
  package's default file is refused unless `overwrite_default=True`. A saved
  library is loaded with `DexonomySynthesizer(cfg, templates=path)`.

```python
dex = DexonomySynthesizer(DexonomyCfg(grow_templates=True))
for obj in objects:
  dex.generate(obj, "five_tip", k=8)
dex.save_templates("my_templates.yaml")
dex = DexonomySynthesizer(DexonomyCfg(), templates="my_templates.yaml")
```

1. **Global alignment** (object frame, batched Adam over all templates). A template contact is
   put on a random surface point, normals opposed, with a random turn about
   the normal. The hand pose is then optimised so that every contact lies on
   the surface against the normal. Filters:
   - alignment error (`max_align_rms`, `max_align_angle`);
   - template-pose penetration (palm 1 mm, fingers 5 mm);
   - contact-force balance carrying the object's weight. Gravity in the
     object frame is `R_obj<-wrist g_wrist`, so it differs per sample.
2. **Distinct poses.** Poses closer than 5 mm and 10° to one already kept
   are dropped.
3. **Refinement and validation.** As in `GraspSynthesizer` below: refine the
   grasping fingers, then the penetration / touch / joint-limit / balance
   filters and the hold test under gravity.

At most `k * attempts_per_sample` grasps are refined and validated, over at
most `max_rounds` alignment rounds.

## GraspSynthesizer (object pose as input)

Finger joint configurations that hold an object at given poses in the hand
base frame.

```python
import numpy as np
from wuji_mjlab.grasp_synth import GraspObject, GraspSynthCfg, GraspSynthesizer

syn = GraspSynthesizer(GraspSynthCfg(
  init_joint_pos={".*_finger1_joint1": 0.8, ".*_finger1_joint2": -0.3, ".*": 0.0},
  hand_quat_w=(0.541675, 0.45452, 0.45452, 0.541675),  # training env wrist: gravity
  attempts_per_sample=10,             # N: at most k * N attempts per pose
))
obj = GraspObject.from_file("mug.obj", mass=0.15)   # or GraspObject.box(half_size)
poses = np.array([[0.004, 0.079, -0.067, 1, 0, 0, 0]])  # (j, 7): pos + wxyz, palm frame
res = syn.generate(obj, poses, "tripod", k=8)
res.joint_pos[0]     # (n <= 8, 20) grasps for pose 0, joint order res.joint_names
res.joint_target[0]  # (n, 20) position targets that hold the object (squeeze)
```

### Inputs and outputs

| Item | Meaning |
|---|---|
| `GraspSynthCfg.init_joint_pos` | Start of the search; fingers not in the grasp type stay here. 20 values or `{regex: value}` |
| `GraspSynthCfg.hand_quat_w` | Palm orientation in the world: the gravity direction in the hold test (default palm up) |
| `obj` | `GraspObject`: mesh (its own frame), mass, convex collision parts (default: the convex hull) |
| `poses` | (j, 7) poses of the mesh frame in `right_palm_link`, m and wxyz |
| `grasp_type` | `pinch`, `tripod`, `five_tip`, `power` (`GRASP_TYPES`) or a custom `GraspType` |
| `k` | Grasps wanted per pose |
| `joint_pos[p]` | (n, 20), n <= k; empty when nothing succeeds within k * N attempts |

The result also has `wrench_residual` per grasp, `attempts` per pose and
`stats`, which counts the candidates that survived each stage.

### Method

Each grasp type is a set of hand contacts: the fingertip pads, plus the palmar
side of the middle and proximal phalanges for `power`. In the refinement
step, every contact is pulled to the nearest object surface point by a spring
and its normal is turned against the object normal (`tau = J^T f + J_rot^T m`).
The object is pinned and gravity is off. MuJoCo contacts with a 1 mm margin
keep the hand just off the surface.

1. **Explore.** Hands start at `init_joint_pos` plus noise on the grasping
   fingers and are refined. A finger that starts inside the object is drawn
   again. The fingers are independent chains on a fixed palm, so every finger
   that ends up touching the object (within 2 mm), without penetrating it and
   inside its joint limits, becomes a candidate.
2. **Combine.** Random combinations of one candidate per finger are scored
   by contact-force balance. The score is the net force and torque of the
   best contact forces inside friction cones (`wrench.py`, Dexonomy's grasp
   quality with g = 0).
3. **Verify.** The best combinations are refined together and checked:
   - No hand-object or hand-hand penetration deeper than 1 mm.
   - Every grasping finger touches the object.
   - Joints stay inside the soft limits (90 % of the range).
   - The actual MuJoCo contacts balance.

   Then comes the hold test with the env's physics. The position actuators
   are set to `q + (J^T f + gravity) / kp`. The object is pinned while the
   grip settles, then released for 1 s, and it must move less than 5 mm and
   10°. With `perturb_force` > 0 the test is repeated with that force along
   ±x, ±y and ±z.

An attempt is one verified combination. A pose stops at k successes or at
k * N attempts. A round in which no balanced combination is found also uses
up the attempts it would have verified.

### Notes

- Refinement ends with a closing phase. The springs are switched off and the
  grasping fingers' distal flexion joints (PIP / DIP; the thumb's MCP / IP)
  curl with `close_torque`. A joint stops once a link it moves touches the
  object, or just short of its soft limit. The other joints hold their
  position. This reaches the fingers that the nearest-point springs leave
  short of the object, e.g. a fingertip just past an edge.
- `power_palm` adds a palm contact to `power`; the palm must touch the
  object. `touch_fraction` < 1 lets only part of the grasping fingers touch
  (the palm still must). It is meant for seeding templates, as for the
  small-diameter rod.

- The hold test includes gravity (`gravity` = 9.81) in the `hand_quat_w`
  orientation, so the grip must also hold the object's weight; 0 only checks
  that the grip alone keeps the object still (net force and torque about 0).
  Two-point grasps (`pinch`) rarely hold the weight: condim 3 contacts have
  no torsional friction, so the object turns about the line between them.
- The simulation uses elliptic friction cones with impratio 10 (the env:
  pyramidal, 1) so that frictional holds do not creep.
- Concave objects need `collision_parts` (a convex decomposition); otherwise
  the convex hull is used for collisions and nearest points.
- The simulation is built per object on first use, which takes a few seconds,
  and is reused for the same `GraspObject`.
