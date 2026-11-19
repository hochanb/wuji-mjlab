"""Neighbour tables over a grasp set for grasp-to-grasp training (DexHOT style).

Each grasp (finger joints, object pose in the palm frame) maps to K hand
keypoints expressed in the object frame: the palm origin and two palm axis
points, and every finger body origin (links 1-4 and the tip pad body). The
distance between two grasps is the mean keypoint displacement [mm]
(DexHOT, arXiv:2611.05119):

  d(c, c') = 1/K sum_j || k_j(c) - k_j(c') ||

so the hand shape and the object pose relative to the palm count together.
For every grasp the table lists the grasps with ``d_min < d <= d_max`` (the
training goals from that grasp), nearest first, up to ``--max-neighbors``.

Usage (from the repo root):
  pixi run python src/tools/grasp_neighbors.py <grasp set folder> [--d-min 20]
      [--d-max 40] [--max-neighbors 64] [--stats-only]

Writes ``<folder>/neighbors.npz``: ``index`` [G, M] int32 (row numbers,
-1 padded), ``count`` [G], ``dist`` [G, M] float32 mm, and the parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

HAND = "wuji_hand2_right"
PALM_AXIS_POINTS = ((0.05, 0.0, 0.0), (0.0, 0.05, 0.0))  # palm frame [m]


def _hand_model() -> tuple[mujoco.MjModel, list[int], list[int]]:
  from wuji_mjlab.grasp_synth.synth import PALM_BODY, hand2_synth_spec
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import hand_joint_limits

  model = hand2_synth_spec().compile()
  names, _, _ = hand_joint_limits(HAND)
  qadr = [int(model.jnt_qposadr[model.joint(n).id]) for n in names]
  bodies = [model.body(PALM_BODY).id] + [
    b
    for b in range(model.nbody)
    if model.body(b).name.startswith("right_finger")
    and (
      model.body(b).name.split("_")[-1].startswith("link")
      or "tip" in model.body(b).name
    )
  ]
  return model, qadr, bodies


def hand_keypoints(joint_q: np.ndarray) -> np.ndarray:
  """[G, K, 3] keypoints in the palm frame for finger joints ``joint_q``."""
  model, qadr, bodies = _hand_model()
  data = mujoco.MjData(model)
  palm = bodies[0]
  out = []
  for q in joint_q:
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    rp, Rp = data.xpos[palm], data.xmat[palm].reshape(3, 3)
    pts = [Rp.T @ (data.xpos[b] - rp) for b in bodies]
    pts += [np.asarray(p) for p in PALM_AXIS_POINTS]
    out.append(pts)
  return np.asarray(out)


def _quat_mat(q: np.ndarray) -> np.ndarray:
  w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ],
    -2,
  )


def object_frame_keypoints(grasps) -> np.ndarray:
  """[G, K, 3] hand keypoints in each grasp's object frame [m]."""
  kp = hand_keypoints(grasps.joint_q.astype(np.float64))
  rot = _quat_mat(grasps.object_quat.astype(np.float64))  # object -> palm
  rel = kp - grasps.object_pos[:, None, :].astype(np.float64)
  return np.einsum("gji,gkj->gki", rot, rel)


def pairwise_distance(kp: np.ndarray, device: str, chunk: int = 512) -> torch.Tensor:
  """[G, G] mean keypoint distance [mm]."""
  k = torch.as_tensor(kp, dtype=torch.float32, device=device) * 1000.0
  g = k.shape[0]
  out = torch.empty(g, g, device=device)
  for s in range(0, g, chunk):
    out[s : s + chunk] = (k[s : s + chunk, None] - k[None]).norm(dim=-1).mean(-1)
  return out


def neighbor_table(d: torch.Tensor, d_min: float, d_max: float, m: int):
  inside = (d > d_min) & (d <= d_max)
  count = inside.sum(-1)
  masked = torch.where(inside, d, torch.full_like(d, float("inf")))
  dist, index = masked.topk(min(m, d.shape[1]), dim=-1, largest=False)
  index = torch.where(torch.isfinite(dist), index, torch.full_like(index, -1))
  return index.int(), count.clamp_max(m).int(), dist, count


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("folder", type=Path, help="grasp set folder (grasps.npz)")
  parser.add_argument("--d-min", type=float, default=20.0, help="mm")
  parser.add_argument("--d-max", type=float, default=40.0, help="mm")
  parser.add_argument("--max-neighbors", type=int, default=64)
  parser.add_argument("--stats-only", action="store_true")
  parser.add_argument(
    "--out", type=Path, default=None, help="table path (default <folder>/neighbors.npz)"
  )
  args = parser.parse_args()

  from wuji_mjlab.tasks.reorient.grasp_stage.schema import load_grasp_set

  grasps = load_grasp_set(args.folder, HAND)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  kp = object_frame_keypoints(grasps)
  d = pairwise_distance(kp, device)
  g = d.shape[0]
  print(f"{args.folder}: {g} grasps, {kp.shape[1]} keypoints")
  off = d + torch.eye(g, device=device) * 1e9
  nn = off.min(-1).values.cpu().numpy()
  print(
    f"  nearest other grasp [mm]: p10/50/90 {np.percentile(nn, [10, 50, 90]).round(1)}"
  )
  allp = (
    d[torch.triu(torch.ones(g, g, dtype=torch.bool, device=device), 1)].cpu().numpy()
  )
  print(
    f"  all pairs [mm]: p1/10/50/90 {np.percentile(allp, [1, 10, 50, 90]).round(1)}"
  )
  for lo, hi in ((0, 20), (20, 40), (40, 70), (20, 70), (70, 1e9)):
    c = ((d > lo) & (d <= hi)).sum(-1).float().cpu().numpy()
    print(
      f"  neighbours in ({lo}, {hi if hi < 1e9 else 'inf'}] mm: per grasp p10/50/90 "
      f"{np.percentile(c, [10, 50, 90]).round(0)}, none: {np.mean(c == 0):.1%}"
    )
  if args.stats_only:
    return
  index, count, dist, full = neighbor_table(
    d, args.d_min, args.d_max, args.max_neighbors
  )
  out = args.out or args.folder / "neighbors.npz"
  np.savez(
    out,
    index=index.cpu().numpy(),
    count=count.cpu().numpy(),
    dist=dist.cpu().numpy().astype(np.float32),
    d_min=args.d_min,
    d_max=args.d_max,
    keypoints=kp.shape[1],
  )
  c = full.cpu().numpy()
  print(
    f"wrote {out}: ({args.d_min}, {args.d_max}] mm, up to {args.max_neighbors} each; "
    f"{np.mean(c == 0):.1%} grasps without a neighbour"
  )


if __name__ == "__main__":
  main()
