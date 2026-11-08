"""Build the grasp-template library (wuji_mjlab.grasp_synth) for every grasp type.

A. Seeds per GRASP taxonomy type (taxonomy.yaml): a sphere / cylinder / box
   sized and placed where the type's starting hand shape holds it (fitted to
   its contacts), the fingers explored from that shape (GraspSynthesizer,
   object pose given, looser seed filters). Validated seeds become templates.
B. Growth: Dexonomy over primitives of several sizes, the stapler and a few
   Objaverse meshes (Dexonomy's processed objaverse_5k, convex pieces), all
   grasp types in one batch per object, grow_templates on.

The library is written to --out (default: the package's templates.yaml).

Usage (repo root):
  pixi run python src/tools/build_grasp_templates.py \\
    [--objaverse tmp/dexonomy_ref/objaverse/objaverse_5k/processed_data] \\
    [--n-objaverse 10] [--epochs 2] [--k 8] [--out ...]
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from wuji_mjlab.grasp_synth import (
  DexonomyCfg,
  DexonomySynthesizer,
  GraspObject,
  GraspSynthCfg,
  GraspSynthesizer,
  load_templates,
)
from wuji_mjlab.grasp_synth.grasp_types import GRASP_TYPES, TAXONOMY
from wuji_mjlab.grasp_synth.templates import (
  TEMPLATES_PATH,
  GraspTemplate,
  save_templates,
  template_from_grasp,
)


def _fit(dex: DexonomySynthesizer, name: str):
  """Sphere fit to the type's starting contacts (centre, radius) and the axis
  the contact normals vary least along (a cylinder axis)."""
  tpl = dex._templates(name, None)[0]
  ctx = dex._ctx(tpl)
  q = torch.tensor(tpl.joint_pos, device=dex.synth.cfg.device)
  p, n = dex._template(q, ctx)
  p, n = (p + 0.005 * n).cpu().numpy(), n.cpu().numpy()  # surface, outward
  a = np.zeros((3 * len(p), 4))
  b = np.zeros(3 * len(p))
  for i in range(len(p)):  # centre = p_i + r n_i
    a[3 * i : 3 * i + 3, :3] = np.eye(3)
    a[3 * i : 3 * i + 3, 3] = -n[i]
    b[3 * i : 3 * i + 3] = p[i]
  x = np.linalg.lstsq(a, b, rcond=None)[0]
  _, vec = np.linalg.eigh(n.T @ n)
  return x[:3], float(np.clip(abs(x[3]), 0.008, 0.06)), vec[:, 0]


def _plate_frame(dex: DexonomySynthesizer, name: str):
  """Plane fit to the fingers' (2-5) starting contacts: a point on the
  contact plane and its normal into the object."""
  tpl = dex._templates(name, None)[0]
  ctx = dex._ctx(tpl)
  q = torch.tensor(tpl.joint_pos, device=dex.synth.cfg.device)
  p, n = dex._template(q, ctx)
  fingers = (ctx.finger >= 2).cpu().numpy()
  p, n = (p + 0.005 * n).cpu().numpy()[fingers], n.cpu().numpy()[fingers]
  normal = n.mean(0)
  return p.mean(0), normal / np.linalg.norm(normal)


def _quat_between(a, b) -> np.ndarray:
  """Quaternion (wxyz) turning unit vector ``a`` onto ``b``."""
  a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
  v, c = np.cross(a, b), float(a @ b)
  if np.linalg.norm(v) < 1e-8:
    if c > 0:
      return np.array([1.0, 0, 0, 0])
    perp = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    return np.array([0.0, *(perp / np.linalg.norm(perp))])
  q = np.array([1.0 + c, *v])
  return q / np.linalg.norm(q)


def _quat_z_to(axis) -> np.ndarray:
  """Quaternion (wxyz) turning +z onto ``axis``."""
  z = np.array([0.0, 0.0, 1.0])
  axis = axis / np.linalg.norm(axis)
  v, c = np.cross(z, axis), float(z @ axis)
  if np.linalg.norm(v) < 1e-8:
    return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
  q = np.array([1.0 + c, *v])
  return q / np.linalg.norm(q)


def _off_palm(
  model, pose: np.ndarray, gap: float = 0.001, touch: bool = False
) -> np.ndarray:
  """Move the object along the palm normal (+y) until it clears the palm by
  ``gap``; with ``touch`` also towards it, so it sits ``gap`` off the palm."""
  import mujoco

  data = mujoco.MjData(model)
  adr = model.joint("object_free").qposadr[0]
  palm = model.geom("right_palm_collision").id
  parts = [
    g for g in range(model.ngeom) if model.geom(g).name.startswith("object_part")
  ]

  def dist(y):
    data.qpos[adr : adr + 7] = (pose[0], y, pose[2], *pose[3:])
    mujoco.mj_kinematics(model, data)
    return min(mujoco.mj_geomDistance(model, data, palm, g, 0.3, None) for g in parts)

  pose = pose.copy()
  if touch:  # bisection on y for distance == gap
    lo, hi = pose[1] - 0.1, pose[1] + 0.1
    for _ in range(40):
      mid = (lo + hi) / 2
      lo, hi = (mid, hi) if dist(mid) < gap else (lo, mid)
    pose[1] = hi
    return pose
  for _ in range(20):
    d = dist(pose[1])
    if d >= gap:
      break
    pose[1] += gap - d + 0.001
  return pose


def seed_templates(
  dex: DexonomySynthesizer,
  k: int,
  names=None,
  poses_per_object: int = 6,
  **cfg,
) -> dict[str, list[GraspTemplate]]:
  """Seeds per type on its matched objects (``TYPE_CATEGORIES``; fitted
  primitives when it has none), placed where the type's start shape holds
  them: half the poses along the fitted axis, half random, off the palm."""
  rng = np.random.default_rng(0)
  out = {}
  for name in names or TAXONOMY:
    centre, r, axis = _fit(dex, name)
    objs = matched_objects(name) or {
      "sphere": GraspObject(
        trimesh.creation.icosphere(subdivisions=3, radius=r), mass=0.1
      ),
      "cylinder": GraspObject(
        trimesh.creation.cylinder(radius=r, height=max(0.06, 3 * r), sections=32),
        mass=0.1,
      ),
    }
    seed_cfg = GraspSynthCfg(**{
      "init_joint_pos": list(dex._templates(name, None)[0].joint_pos),
      "explore_noise": 0.3,
      "explore_worlds": 256,
      "attempts_per_sample": 5,
      # Seeds only start templates (Dexonomy then validates grasps fully).
      "touch_fraction": 0.5,
      "max_wrench_residual": 0.5,
      "max_pos_drift": 0.02,
      "max_rot_drift": np.radians(30.0),
      **cfg,
    })  # fmt: skip
    syn = GraspSynthesizer(seed_cfg)
    fingers = {c.finger for c in GRASP_TYPES[name].contacts}
    found = []
    t0 = time.time()
    for obj in objs.values():
      model = syn._build_model(obj)
      base = _quat_z_to(axis)
      poses = []
      extents = np.asarray(obj.mesh.extents)
      plate = extents.min() < 0.3 * np.sort(extents)[1]
      for i in range(poses_per_object):
        if plate:  # thin plate: its face on the fingers' contact plane
          point, normal = _plate_frame(dex, name)
          thin = np.eye(3)[int(np.argmin(extents))]
          twist = trimesh.transformations.quaternion_about_axis(
            rng.uniform(0, 2 * np.pi), normal
          )
          rot = trimesh.transformations.quaternion_multiply(
            twist, _quat_between(thin, normal)
          )
          centre_i = point + normal * (extents.min() / 2 + 0.0005)
          poses.append(np.array([*(centre_i + rng.normal(scale=0.002, size=3)), *rot]))
          continue
        if i % 2 == 0:  # object z along the fitted axis, random turn about it
          q = trimesh.transformations.quaternion_about_axis(
            rng.uniform(0, 2 * np.pi), axis
          )
          rot = trimesh.transformations.quaternion_multiply(q, base)
        else:
          rot = trimesh.transformations.random_quaternion(rng.random(3))
        pose = np.array([*(centre + rng.normal(scale=0.003, size=3)), *rot])
        on_palm = 0 in fingers  # palm contact: sit against the palm
        poses.append(_off_palm(model, pose, 0.0005 if on_palm else 0.001, on_palm))
      res = syn.generate(obj, np.array(poses), name, k)
      for p, qs in enumerate(res.joint_pos):
        for q in qs:
          tpl = template_from_grasp(model, name, q, np.array(poses[p]))
          contacts = tuple(c for c in tpl.contacts if c.finger in fingers)
          if {c.finger for c in contacts} >= (fingers - {0}) and len(contacts) >= 2:
            found.append(GraspTemplate(name, tpl.joint_pos, contacts))
    out[name] = found
    print(
      f"seed {name}: {len(objs)} objects, {len(found)} templates ({time.time() - t0:.0f} s)",
      flush=True,
    )
  return out


# Hand-made types that are GRASP taxonomy types: their templates seed those.
ALIASES = {
  "1_large_diameter": ("large_diameter",),
  "3_medium_wrap": ("power_palm",),
  "4_adducted_thumb": ("adducted_thumb",),
  "9_palmar_pinch": ("pinch",),
  "10_power_disk": ("power_palm",),
  "11_power_sphere": ("power",),
  "13_precision_sphere": ("five_tip",),
  "14_tripod": ("tripod",),
  "24_tip_pinch": ("pinch",),
}


def _cyl(d_mm, length_mm):
  return GraspObject(
    trimesh.creation.cylinder(radius=d_mm / 2000, height=length_mm / 1000, sections=32),
    mass=0.1,
  )


def _box(x_mm, y_mm, z_mm, mass=0.1):
  return GraspObject.box((x_mm / 2000, y_mm / 2000, z_mm / 2000), mass=mass)


def _sphere(d_mm):
  return GraspObject(
    trimesh.creation.icosphere(subdivisions=3, radius=d_mm / 2000), mass=0.1
  )


# Objects by category (sizes in mm): what each grasp type is tried on first.
CATALOG = {
  "small": {
    "sphere12": lambda: _sphere(12),
    "cube14": lambda: _box(14, 14, 14),
    "pin8x40": lambda: _cyl(8, 40),
    "bead18": lambda: _sphere(18),
  },
  "pen": {"pen9x140": lambda: _cyl(9, 140), "pen12x130": lambda: _cyl(12, 130)},
  "handle": {
    "rod20x120": lambda: _cyl(20, 120),
    "rod26x130": lambda: _cyl(26, 130),
    "bar24x30x130": lambda: _box(24, 30, 130),
    "stapler": lambda: GraspObject.from_compound("stapler"),
  },
  "cyl_mid": {"cyl35x100": lambda: _cyl(35, 100), "cyl45x100": lambda: _cyl(45, 100)},
  "cyl_large": {"cyl55x110": lambda: _cyl(55, 110), "cyl70x120": lambda: _cyl(70, 120)},
  "disk": {"disk65x20": lambda: _cyl(65, 20), "disk85x25": lambda: _cyl(85, 25)},
  "sphere_s": {"sphere30": lambda: _sphere(30), "sphere40": lambda: _sphere(40)},
  "sphere_m": {"sphere55": lambda: _sphere(55), "sphere65": lambda: _sphere(65)},
  "sphere_l": {"sphere75": lambda: _sphere(75)},
  "cube": {"cube30": lambda: _box(30, 30, 30), "cube45": lambda: _box(45, 45, 45)},
  "card": {
    "card3x40x60": lambda: _box(3, 40, 60, 0.03),
    "key5x25x50": lambda: _box(5, 25, 50, 0.03),
  },
  "book": {
    "book25x100x140": lambda: _box(25, 100, 140, 0.25),
    "book35x90x130": lambda: _box(35, 90, 130, 0.25),
  },
  "plate": {"plate12x70x100": lambda: _box(12, 70, 100, 0.12)},
  "thin_plate": {
    "thinplate6x60x90": lambda: _box(6, 60, 90, 0.06),
    "thinplate8x70x110": lambda: _box(8, 70, 110, 0.08),
    "thinplate10x60x100": lambda: _box(10, 60, 100, 0.08),
  },
}
TYPE_CATEGORIES = {
  "1_large_diameter": ("cyl_large",), "large_diameter": ("cyl_large",),
  "2_small_diameter": ("handle",), "small_diameter": ("handle",),
  "3_medium_wrap": ("cyl_mid",), "power_palm": ("cyl_mid", "sphere_m"),
  "4_adducted_thumb": ("handle",), "adducted_thumb": ("handle",),
  "5_light_tool": ("pen",), "6_prismatic_4_finger": ("pen", "handle"),
  "7_prismatic_3_finger": ("pen", "small"), "8_prismatic_2_finger": ("pen", "small"),
  "9_palmar_pinch": ("small", "card"), "10_power_disk": ("disk",),
  "11_power_sphere": ("sphere_m", "sphere_l"), "power": ("sphere_m", "cyl_mid"),
  "12_precision_disk": ("disk",), "13_precision_sphere": ("sphere_s", "sphere_m"),
  "14_tripod": ("small", "sphere_s", "cube"), "tripod": ("small", "sphere_s", "cube"),
  "15_fixed_hook": ("handle",), "16_lateral": ("card",), "17_index_finger_extension": ("handle",),
  "18_extension_type": ("book", "plate"), "19_distal": ("pen",), "20_writing_tripod": ("pen",),
  "21_tripod_variation": ("small", "pen"), "22_parallel_extension": ("thin_plate",),
  "23_adduction_grip": ("pen", "small"), "24_tip_pinch": ("small",), "pinch": ("small", "cube"),
  "25_lateral_tripod": ("small", "pen"), "26_sphere_4_finger": ("sphere_m",),
  "27_quadpod": ("sphere_s", "cube"), "28_sphere_3_finger": ("sphere_m",), "five_tip": ("sphere_s", "cube"),
  "29_stick": ("handle",), "30_palmar": ("thin_plate",), "31_ring": ("cyl_mid", "handle"),
  "32_ventral": ("pen",), "33_inferior_pincer": ("small", "sphere_s"),
}  # fmt: skip


def matched_objects(name: str) -> dict[str, GraspObject]:
  return {
    k: make()
    for cat in TYPE_CATEGORIES.get(name, ())
    for k, make in CATALOG[cat].items()
  }


def clean_library(
  library: dict[str, list[GraspTemplate]], joint_names
) -> dict[str, int]:
  """Drop templates outside their type's joint bounds, and small-diameter
  ones without the thumb (it wraps from the other side). Returns the
  number dropped per type."""
  dropped = {}
  for name, items in library.items():
    gtype = GRASP_TYPES.get(name)
    keep = []
    for t in items:
      ok = True
      if gtype is not None and gtype.joint_bounds:
        lo, hi = gtype.bounds(joint_names, np.full(20, -9.0), np.full(20, 9.0))
        q = np.asarray(t.joint_pos)
        ok = bool(np.all(q >= lo - 0.02) and np.all(q <= hi + 0.02))
      if name == "2_small_diameter" and 1 not in {c.finger for c in t.contacts}:
        ok = False
      if ok:
        keep.append(t)
    dropped[name] = len(items) - len(keep)
    library[name] = keep
  return dropped


def add_aliases(library: dict[str, list[GraspTemplate]]) -> None:
  for target, sources in ALIASES.items():
    have = library.setdefault(target, [])
    for src in sources:
      for t in library.get(src, []):
        if len(have) < 64:
          have.append(GraspTemplate(target, t.joint_pos, t.contacts))


def objects(objaverse: str | None, n_objaverse: int) -> dict[str, GraspObject]:
  objs = {}
  for d in (30, 45, 60):
    objs[f"cube{d}"] = GraspObject.box((d / 2000,) * 3, mass=0.1)
  for d in (30, 45, 60, 75):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=d / 2000)
    objs[f"sphere{d}"] = GraspObject(mesh, mass=0.1)
  for d, length in ((20, 120), (35, 100), (50, 100), (70, 120)):
    mesh = trimesh.creation.cylinder(radius=d / 2000, height=length / 1000, sections=32)
    objs[f"cylinder{d}x{length}"] = GraspObject(mesh, mass=0.12)
  objs["plate8x60x90"] = GraspObject.box((0.004, 0.03, 0.045), mass=0.05)
  objs["box40x60x100"] = GraspObject.box((0.02, 0.03, 0.05), mass=0.15)
  objs["stapler"] = GraspObject.from_compound("stapler")
  if objaverse:
    rng = np.random.default_rng(1)
    dirs = sorted(
      p for p in Path(objaverse).iterdir() if (p / "urdf" / "meshes").is_dir()
    )
    for d in rng.permutation(dirs)[: n_objaverse * 3]:
      pieces = sorted(glob.glob(str(d / "urdf" / "meshes" / "convex_piece_*.obj")))
      if not pieces or len(objs) >= 14 + n_objaverse:
        continue
      try:
        mesh = trimesh.load(str(d / "mesh" / "simplified.obj"), force="mesh")
        parts = [trimesh.load(p, force="mesh") for p in pieces]
        # Many small pieces overflow the contact buffers: keep the 12 largest.
        parts = sorted(parts, key=lambda m: -m.convex_hull.volume)[:12]
      except Exception:  # noqa: BLE001 - skip unreadable meshes
        continue
      size = rng.uniform(0.06, 0.14)  # longest extent [m]
      scale = size / float(np.max(mesh.extents))
      mesh.apply_scale(scale)
      for p in parts:
        p.apply_scale(scale)
      objs[f"objaverse_{d.name[:8]}"] = GraspObject(
        mesh, mass=0.12, collision_parts=parts
      )
  return objs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument(
    "--objaverse", default="tmp/dexonomy_ref/objaverse/objaverse_5k/processed_data"
  )
  parser.add_argument("--n-objaverse", type=int, default=10)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--k", type=int, default=8)
  parser.add_argument("--out", default=str(TEMPLATES_PATH))
  parser.add_argument(
    "--start", default=None, help="library to start from (default: --out)"
  )
  parser.add_argument("--skip-seeds", action="store_true")
  parser.add_argument(
    "--seed-types", nargs="*", default=None, help="default: all taxonomy types"
  )
  parser.add_argument("--add", nargs="*", default=[], help="more libraries to merge in")
  parser.add_argument(
    "--growth-categories", nargs="*", default=None, help="grow on these categories only"
  )
  parser.add_argument("--seed-poses", type=int, default=6, help="poses per object")
  parser.add_argument(
    "--objaverse-epoch",
    action="store_true",
    help="after the matched objects, every type on Objaverse",
  )
  args = parser.parse_args()

  start = load_templates(args.start) if args.start else load_templates(args.out)
  dex = DexonomySynthesizer(
    DexonomyCfg(
      grow_templates=True,
      max_rounds=10,
      synth=GraspSynthCfg(nconmax=256, njmax=2000),  # multi-part meshes
    ),
    templates=start,
  )
  dex.synth._setup(GraspObject.box((0.02,) * 3), 64)
  for path in args.add:
    for name, items in load_templates(path).items():
      dex.library.setdefault(name, []).extend(items)
  dropped = clean_library(dex.library, dex.synth.joint_names)
  print("cleaned:", {k: v for k, v in dropped.items() if v}, flush=True)
  if not args.skip_seeds:
    seeds = seed_templates(
      dex, 2, names=args.seed_types, poses_per_object=args.seed_poses
    )
    for name, items in seeds.items():
      if items:
        dex.library.setdefault(name, []).extend(items)
  add_aliases(dex.library)
  clean_library(dex.library, dex.synth.joint_names)
  save_templates(dex.library, args.out)

  # Growth: every object of the catalog with the grasp types matched to it.
  plan = []
  for cat, makers in CATALOG.items():
    if args.growth_categories is not None and cat not in args.growth_categories:
      continue
    types = [t for t, cats in TYPE_CATEGORIES.items() if cat in cats]
    plan += [(name, make, types) for name, make in makers.items()]
  if args.objaverse_epoch:
    plan += [
      (name, (lambda o=o: o), None)
      for name, o in objects(args.objaverse, args.n_objaverse).items()
      if name.startswith("objaverse_")
    ]
  print(f"growth: {len(plan)} objects", flush=True)
  for epoch in range(args.epochs):
    for oname, make, types in plan:
      have = [t for t in (types or dex.library) if dex.library.get(t)]
      if not have:
        continue
      t0 = time.time()
      res = dex.generate_many(make(), have, args.k)
      got = sum(1 for r in res.values() if len(r.joint_pos))
      print(
        f"epoch {epoch} {oname}: {got}/{len(have)} types with grasps, "
        f"{sum(len(r.joint_pos) for r in res.values())} grasps ({time.time() - t0:.0f} s), "
        f"library {sum(len(v) for v in dex.library.values())}",
        flush=True,
      )
      save_templates(dex.library, args.out)
  print({t: len(v) for t, v in dex.library.items()}, flush=True)


if __name__ == "__main__":
  main()
