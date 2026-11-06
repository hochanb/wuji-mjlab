"""Random object-pose grasp sets (no hand shape) for testing the grasp-stage task.

Temporary helper, separate from the training pipeline: it imitates the
reorient pose-goal task with grasp sets. For an object it writes N initial
and N goal "grasps" that only vary the object pose; the hand part is off
(tip / joint weights 0), and the initial grasps keep the hand in its usual
reset state.

- init: the object at its usual start (the reorient cube's spawn point, plus
  ``--lift``) with +-``--pos-noise`` per world axis and a random orientation;
  finger joints drawn from the reorient reset ranges.
- goal: position uniform in the reorient goal box (wrist-tag frame), a random
  orientation, pose weights 1 (both required).

Orientations are uniform within a maximum angle of the object's default
start orientation (its spawn orientation at a reorient reset), the same
reference for init and goal, so the rotation a stage asks for is at most
twice that angle; 180 deg is uniform on SO(3).

- single set (default): uniform on SO(3); writes
  ``<object-dir>/grasps/<name>/{init,goal}/grasps.npz``.
- ``--stages K``: a curriculum of K sets ``<name>_stage1`` .. ``_stageK`` whose
  maximum angle grows evenly to 180 deg (``--max-angles`` to set them).

``--training`` also writes the training file next to the objects folder:
``trainings/<name>.yaml`` (one stage) or ``trainings/<name>_curriculum.yaml``
(K stages, ``--iterations`` each).

Usage (repo root):
  pixi run python src/tools/make_pose_grasps.py \\
    --object-dir src/wuji_mjlab/assets/grasp_stage/examples/cube_6d/objects/cube \\
    --name cube_6d_10k --count 10000 [--seed 0] [--stages 5] [--training]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

HAND = "wuji_hand2_right"
# Reorient pose-goal box in the wrist-tag frame [m] (see _GOAL_POS_RANGE).
GOAL_BOX = ((-0.179, -0.143), (-0.006, 0.019), (-0.006, 0.018))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * np.cross(xyz, v)
  return v + w * t + np.cross(xyz, t)


def _uniform_quats(rng: np.random.Generator, n: int) -> np.ndarray:
  """Uniform on SO(3) (wxyz, w >= 0)."""
  q = rng.normal(size=(n, 4))
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def _quats_near(rng, n: int, ref: np.ndarray, max_angle: float) -> np.ndarray:
  """``ref * delta``, delta uniform on SO(3) restricted to rotations of at most
  ``max_angle`` [rad] (rejection sampling); uniform on SO(3) at pi."""
  out, have = [], 0
  while have < n:
    q = _uniform_quats(rng, max(4 * (n - have), 1024))
    q = q[2.0 * np.arccos(np.clip(q[:, 0], -1.0, 1.0)) <= max_angle + 1e-9]
    out.append(q)
    have += len(q)
  delta = np.concatenate(out)[:n]
  q = _quat_mul(np.broadcast_to(ref, (n, 4)), delta)
  return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def _frames():
  """Palm pose, wrist tag pose and the object's spawn point in the world at a
  reorient reset (deterministic: fixed mount pitch, wrist at zero), plus the
  finger joint names and their reset ranges."""
  import wuji_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("WujiHand2_ReorientPosWrist_50Hz", play=True)
  cfg.scene.num_envs = 1
  ranges = cfg.events["reset_robot_joints"].params["position_range"]
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  env.sim.forward()
  cmd = env.command_manager.get_term("reorient_command")
  robot, obj = cmd.robot, cmd.object
  palm = robot.data.body_link_pose_w[0, cmd.palm_body_id].numpy().astype(np.float64)
  tag = robot.data.site_pose_w[0, cmd.tag_site_id].numpy().astype(np.float64)
  spawn = (obj.data.default_root_state[0, :3] + env.scene.env_origins[0]).numpy()
  _, names = robot.find_joints(".*_finger.*_joint.*")
  env.close()
  lo_hi = []
  for name in names:
    match = [r for pattern, r in ranges.items() if re.fullmatch(pattern, name)]
    lo_hi.append(match[0] if match else (0.0, 0.0))
  return palm, tag, spawn.astype(np.float64), list(names), np.array(lo_hi)


def make_sets(
  frames, count: int, seed: int, lift: float, pos_noise: float, max_angle: float
):
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import GraspSet

  rng = np.random.default_rng(seed)
  palm, tag, spawn, _, joint_ranges = frames
  palm_p, palm_q = palm[:3], palm[3:]
  palm_q_inv = palm_q * np.array([1.0, -1.0, -1.0, -1.0])
  # Tag pose in the palm frame (fixed).
  tag_p = _quat_rotate(palm_q_inv, tag[:3] - palm_p)
  tag_q = _quat_mul(palm_q_inv, tag[3:])
  # Default start orientation (identity in the world at a reset), palm frame.
  ref = palm_q_inv

  def hand_off(n: int) -> dict:
    return {
      "tip_points": np.zeros((n, 5, 3)),
      "tip_normals": np.tile([0.0, 0.0, 1.0], (n, 5, 1)),
      "tip_weights": np.zeros((n, 5)),
      "joint_weights": np.zeros((n, 5)),
    }

  # Initial: spawn point (+ lift, +- noise per world axis), random orientation,
  # finger joints from the reset ranges.
  noise = rng.uniform(-pos_noise, pos_noise, size=(count, 3))
  pos_w = spawn + np.array([0.0, 0.0, lift]) + noise
  init = GraspSet(
    np.arange(count),
    _quat_rotate(np.broadcast_to(palm_q_inv, (count, 4)), pos_w - palm_p),
    _quats_near(rng, count, ref, max_angle),
    np.ones((count, 2)),
    joint_q=rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1], size=(count, 20)),
    **hand_off(count),
  )
  # Goal: position uniform in the tag-frame box, random orientation.
  box = np.array(GOAL_BOX)
  pos_tag = rng.uniform(box[:, 0], box[:, 1], size=(count, 3))
  goal = GraspSet(
    np.arange(count),
    tag_p + _quat_rotate(np.broadcast_to(tag_q, (count, 4)), pos_tag),
    _quats_near(rng, count, ref, max_angle),
    np.ones((count, 2)),
    joint_q=np.broadcast_to(joint_ranges.mean(1), (count, 20)).copy(),
    **hand_off(count),
  )
  return init, goal


def _training_yaml(obj: str, stages: list[tuple[str, str]], iterations: int) -> str:
  body = "".join(
    f"""  - name: {stage}
    objects: [{obj}]
    init_grasps: {{{obj}: [grasps/{folder}/init]}}
    goal_grasps: {{{obj}: [grasps/{folder}/goal]}}
    advance: {{after_iterations: {iterations}}}
"""
    for stage, folder in stages
  )
  return f"""# {obj} 6D pose task from make_pose_grasps.py (no hand shape): random
# initial poses at the spawn point, random goal poses in the goal box.
version: 1
hand: {HAND}
encoding: {{name: none, dim: 0}}
object_root: ../objects
stages:
{body}"""


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--object-dir", required=True, type=Path)
  parser.add_argument("--name", required=True)
  parser.add_argument("--count", required=True, type=int)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--lift", type=float, default=0.0, help="spawn height offset [m]")
  parser.add_argument("--pos-noise", type=float, default=0.01, help="spawn noise [m]")
  parser.add_argument(
    "--stages", type=int, default=0, help="K > 0: <name>_stage1..K curriculum"
  )
  parser.add_argument(
    "--max-angles", default=None, help="comma-separated max angles [deg] per stage"
  )
  parser.add_argument(
    "--training", action="store_true", help="also write the training file"
  )
  parser.add_argument(
    "--iterations", type=int, default=None, help="per stage (5000 single, 1000 staged)"
  )
  args = parser.parse_args()

  from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
    load_grasp_set,
    load_training,
    save_grasp_set_npz,
  )

  if args.stages > 0:
    if args.max_angles:
      angles = [float(a) for a in args.max_angles.split(",")]
      if len(angles) != args.stages:
        raise SystemExit(f"--max-angles needs {args.stages} values")
    else:
      angles = [180.0 * (k + 1) / args.stages for k in range(args.stages)]
    sets = [(f"{args.name}_stage{k + 1}", a) for k, a in enumerate(angles)]
  else:
    sets = [(args.name, 180.0)]
  frames = _frames()
  for k, (folder, angle) in enumerate(sets):
    init, goal = make_sets(
      frames, args.count, args.seed + k, args.lift, args.pos_noise, np.radians(angle)
    )
    root = args.object_dir / "grasps" / folder
    for role, grasps in (("init", init), ("goal", goal)):
      path = save_grasp_set_npz(root / role, grasps, HAND)
      n = len(load_grasp_set(root / role, HAND))  # validates
      print(
        f"{path}: {n} grasps, max angle {angle:g} deg, {path.stat().st_size / 1e6:.1f} MB"
      )
  if args.training:
    obj = args.object_dir.name
    trainings = args.object_dir.parent.parent / "trainings"
    trainings.mkdir(exist_ok=True)
    staged = args.stages > 0
    iterations = args.iterations or (1000 if staged else 5000)
    path = trainings / (
      f"{args.name}_curriculum.yaml" if staged else f"{args.name}.yaml"
    )
    stages = [
      (f"stage{k + 1}" if staged else "pose", folder)
      for k, (folder, _) in enumerate(sets)
    ]
    path.write_text(_training_yaml(obj, stages, iterations))
    load_training(path)  # validates every stage, incl. joint limits
    print(f"{path}: written ({len(stages)} stage(s), {iterations} iterations each)")


if __name__ == "__main__":
  main()
