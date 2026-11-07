# Grasp synthesis (Wuji Hand 2)

Finger joint configurations that hold an object at given poses in the hand
base frame. The method follows the simulation stages of Dexonomy
([arXiv:2504.18829](https://arxiv.org/abs/2504.18829)). Every attempt is one
mujoco-warp world, so all poses run in parallel on one GPU.

```python
import numpy as np
from wuji_mjlab.grasp_synth import GraspObject, GraspSynthCfg, GraspSynthesizer

syn = GraspSynthesizer(GraspSynthCfg(
  init_joint_pos={".*_finger1_joint1": 0.8, ".*_finger1_joint2": -0.3, ".*": 0.0},
  hand_quat_w=(0.5, 0.5, 0.5, 0.5),   # palm up (reorient env): gravity direction
  attempts_per_sample=10,             # N: at most k * N attempts per pose
))
obj = GraspObject.from_file("mug.obj", mass=0.15)   # or GraspObject.box(half_size)
poses = np.array([[0.004, 0.079, -0.067, 1, 0, 0, 0]])  # (j, 7): pos + wxyz, palm frame
res = syn.generate(obj, poses, "tripod", k=8)
res.joint_pos[0]     # (n <= 8, 20) grasps for pose 0, joint order res.joint_names
res.joint_target[0]  # (n, 20) position targets that hold the object (squeeze)
```

## Inputs and outputs

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

## Method

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

## Notes

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
