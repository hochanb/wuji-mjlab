# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Memory-bounded Simulation facade: runs physics in fixed-size chunks.

The dominant GPU memory cost of a large vectorized MuJoCo Warp simulation is
not the persistent per-world state (qpos, qvel, contacts, ...) -- that scales
gently with num_envs. It is the *transient* collision scratch inside
mujoco_warp's EPA/GJK narrowphase (``epa_face``, ``epa_pr``, ...), sized by
``naccdmax`` (~ nconmax_per_world * num_envs) and ``ccd_iterations``. On a
CUDA-graph-backed Simulation, that scratch gets permanently pinned in the
graph's private memory pool on the graph's first launch -- see the
investigation this module grew out of.

``ChunkedSimulation`` keeps ONE small ("worker") Simulation whose CUDA graphs
are sized for ``chunk_size`` envs (bounding that scratch to a constant,
independent of the logical total), and a ("master") Simulation sized for the
full ``num_envs`` that is NEVER stepped directly -- it only holds the
persistent per-world buffers that the rest of mjlab (Scene, managers, DR
events) read and write via ``sim.data`` / ``sim.model``, exactly as today.

Every public operation (`step`, `forward`, `sense`, `reset`,
`expand_model_fields`, ...) copies the relevant chunk's persistent fields
master -> worker, runs the real (chunk-sized, graph-backed) operation on the
worker, then copies the results worker -> master. The copy is a brute-force
"copy every per-world array field, both directions, every call" -- simple to
reason about, and cheap: the persistent fields this copies are the ~100 MiB
side of the problem, not the multi-GiB transient scratch (which never leaves
the worker).

Known gaps (not needed by the reorient task today, left for follow-up):
  - Mesh/geom *variants* (``variant_info``): the worker is built with the same
    ``variant_info`` as the master, which is only correct when it is empty
    (no per-world mesh variants). Raises at construction otherwise.
  - Sensor context (camera/raycast ``sense()``): only wired up when
    ``sensor_context is None`` (true for the reorient task). Raises otherwise.
  - The viser interactive viewer asserts ``isinstance(sim, Simulation)``;
    ``ChunkedSimulation`` is a standalone facade, not a subclass, so it is
    not compatible with ``--viewer viser``. Training (``train_rsl_rl.py``)
    never touches the viewer, so this is out of scope for now.
  - ``step()``/``forward()`` are fused into one captured CUDA graph each
    (see ``_capture_fused``), calling ``mjwarp.step``/``mjwarp.forward``
    directly instead of going through ``Simulation.step()``/``.forward()``.
    This skips ``Simulation``'s own ``nan_guard`` wrapper; NaN-guard
    diagnostics (``cfg.sim.nan_guard.enabled``, off by default) are not
    available through a fused ChunkedSimulation.

Wiring it in: call ``install_chunked_simulation(chunk_size)`` once, before any
``ManagerBasedRlEnv`` is constructed, to monkeypatch the ``Simulation`` symbol
that ``mjlab.envs.manager_based_rl_env`` resolves at call time. In practice
this is driven by the ``WUJI_SIM_CHUNK_SIZE`` env var, checked once at
``wuji_mjlab.tasks`` import time (see ``wuji_mjlab/tasks/__init__.py``), so
every entry point that does ``import wuji_mjlab.tasks`` picks it up without
further changes.
"""

from __future__ import annotations

import functools
import importlib.metadata
import warnings

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.sim import Simulation, SimulationCfg

_SKIP_FIELDS = frozenset({"nworld", "nconmax", "naconmax", "naccdmax", "nccdmax", "njmax"})


def _iter_world_array_fields(struct, nworld: int):
  """Yield (name, wp.array) for array fields whose first axis is truly world-sized.

  Some Model/Data fields are shared/broadcast across worlds (shape[0] == 1,
  e.g. a model field nothing has expanded for per-env DR yet) -- those are
  identical on master and worker by construction (both come from the same
  compiled spec) and must NOT be sliced as if indexed by world.
  """
  for name in dir(struct):
    if name.startswith("_") or name in _SKIP_FIELDS:
      continue
    try:
      val = getattr(struct, name)
    except Exception:
      continue
    if isinstance(val, wp.array) and len(val.shape) > 0 and val.shape[0] == nworld:
      yield name, val


def _build_copy_pairs(src, dst, src_nworld: int, dst_nworld: int) -> list[tuple]:
  """Precompute the (src_tensor, dst_tensor) views for every matching per-world field.

  Benchmarking showed the naive version of this (re-running ``dir()`` /
  ``getattr()`` / ``isinstance()`` over ~130 struct attributes, then
  re-wrapping each with ``wp.to_torch()``) on *every single chunk launch* is
  the dominant per-step cost -- constant ~5.5 ms/launch regardless of chunk
  size (256 vs. 1024), i.e. pure Python/reflection overhead, not actual
  memory-bandwidth-bound copying. Doing this reflection once per
  Simulation/field-set lifetime (construction, and again whenever
  ``expand_model_fields`` replaces model arrays) and caching the resulting
  torch views turns the hot path into a plain list of tensor slice-assigns.
  """
  pairs = []
  for name, src_arr in _iter_world_array_fields(src, src_nworld):
    dst_arr = getattr(dst, name, None)
    if not isinstance(dst_arr, wp.array) or dst_arr.shape[0] != dst_nworld:
      continue
    if dst_arr.shape[1:] != src_arr.shape[1:] or dst_arr.dtype != src_arr.dtype:
      # Shape/dtype mismatch: not actually a per-world twin of this field
      # (e.g. a field that exists on one struct but means something
      # differently-shaped on the other). Skip rather than guess.
      continue
    pairs.append((wp.to_torch(src_arr, requires_grad=False), wp.to_torch(dst_arr, requires_grad=False)))
  return pairs


def _apply_copy_pairs(pairs: list[tuple], src_idx, dst_idx) -> None:
  for src_t, dst_t in pairs:
    dst_t[dst_idx] = src_t[src_idx]


class _NoGraphSimulation(Simulation):
  """A Simulation that never captures/launches CUDA graphs.

  Used for the master (full num_envs) instance: construction and
  ``expand_model_fields`` must stay cheap and safe at any num_envs, and this
  instance's own step/forward/sense/reset are never called (ChunkedSimulation
  routes all physics through the worker), so graph capture would be pure
  waste -- and, at very large num_envs, is exactly the thing that can OOM
  during capture itself.
  """

  def _should_use_cuda_graph(self) -> bool:
    return False


class ChunkedSimulation:
  """Drop-in replacement for ``mjlab.sim.Simulation`` with bounded physics memory.

  Construct with the same arguments as ``Simulation``, plus ``chunk_size``.
  ``num_envs`` must be a multiple of ``chunk_size``.
  """

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    chunk_size: int,
    model: mujoco.MjModel | None = None,
    device: str = "cuda:0",
    *,
    spec: mujoco.MjSpec | None = None,
    variant_info: list | None = None,
  ) -> None:
    if num_envs % chunk_size != 0:
      raise ValueError(f"num_envs ({num_envs}) must be a multiple of chunk_size ({chunk_size})")
    if variant_info:
      raise NotImplementedError(
        "ChunkedSimulation does not yet support per-world mesh variants "
        "(variant_info); the worker would need a chunk-local variant "
        "assignment. Not needed by the reorient task today."
      )

    self.num_envs = num_envs
    self.chunk_size = chunk_size
    self.num_chunks = num_envs // chunk_size
    self.device = device

    self._master = _NoGraphSimulation(
      num_envs=num_envs, cfg=cfg, model=model, device=device, spec=spec, variant_info=variant_info
    )
    self._worker = Simulation(
      num_envs=chunk_size, cfg=cfg, model=model, device=device, spec=spec, variant_info=variant_info
    )
    # Data arrays are stable for the Simulation's lifetime (nothing replaces
    # them the way expand_model_fields() replaces model arrays), so these
    # are built once and never invalidated.
    self._data_in_pairs = _build_copy_pairs(
      self._master.wp_data, self._worker.wp_data, num_envs, chunk_size
    )
    self._data_out_pairs = _build_copy_pairs(
      self._worker.wp_data, self._master.wp_data, chunk_size, num_envs
    )
    self._rebuild_model_pairs()

    # step()/forward() are captured lazily (on first use) as ONE graph each,
    # covering the whole "for chunk: copy_in; mjwarp op; copy_out" loop --
    # see _capture_fused. Both touch every chunk unconditionally on every
    # call (no data-dependent skipping), so fusing them has no downside.
    # reset() is deliberately NOT fused: it skips chunks with no env_ids in
    # them, which is a real, data-dependent win on the common sparse-reset
    # case that a static graph would have to give up (always processing
    # every chunk). Left as the plain per-chunk Python loop.
    self._step_graph = None
    self._forward_graph = None
    # Torch ops inside our capture must run on warp's own capturing stream
    # (TorchArray does this per-op already; we do it once for the whole
    # fused region instead). Same pattern as TorchArray._setup_stream().
    self._torch_stream = torch.cuda.ExternalStream(
      wp.get_stream(self._worker.wp_device).cuda_stream
    )

  def _rebuild_model_pairs(self) -> None:
    """(Re)build the cached model-field copy pairs.

    Must be called after any model-array replacement -- i.e. construction,
    and again at the end of ``expand_model_fields`` (mirrors
    ``Simulation.create_graph()``'s own "replaced arrays -> must recapture"
    rule, just for our copy cache instead of CUDA graphs).
    """
    self._model_in_pairs = _build_copy_pairs(
      self._master.wp_model, self._worker.wp_model, self.num_envs, self.chunk_size
    )

  def _capture_fused(self, mjwarp_fn) -> "wp.Graph":
    """Capture the whole chunk loop for one mjwarp op as a single CUDA graph.

    Calls the raw ``mjwarp.step``/``mjwarp.forward`` function directly
    (bypassing ``Simulation.step()``/``.forward()``'s own graph machinery
    and nan_guard) so there is no nested graph-launch to reason about: this
    capture owns the entire sequence of GPU work itself.
    """
    with wp.ScopedDevice(self._worker.wp_device):
      with torch.cuda.stream(self._torch_stream):
        with wp.ScopedCapture() as capture:
          for i in range(self.num_chunks):
            chunk = self._chunk_slice(i)
            self._copy_in(chunk)
            mjwarp_fn(self._worker.wp_model, self._worker.wp_data)
            self._copy_out(chunk)
    return capture.graph

  # -- Pass-through surface (storage lives on the master; never stepped). --

  @property
  def mj_model(self):
    return self._master.mj_model

  @property
  def mj_data(self):
    return self._master.mj_data

  @property
  def wp_model(self):
    return self._master.wp_model

  @property
  def wp_data(self):
    return self._master.wp_data

  @property
  def data(self):
    return self._master.data

  @property
  def model(self):
    return self._master.model

  @property
  def default_model_fields(self):
    return self._master.default_model_fields

  @property
  def expanded_fields(self):
    return self._master.expanded_fields

  @property
  def per_world_default_fields(self):
    return self._master.per_world_default_fields

  @property
  def world_to_variant(self):
    return self._master.world_to_variant

  def get_default_field(self, field: str) -> torch.Tensor:
    return self._master.get_default_field(field)

  def set_sensor_context(self, ctx) -> None:
    raise NotImplementedError("ChunkedSimulation does not yet support a sensor context.")

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    if not fields:
      return
    # Master: storage only, cheap no-op graph recreation (_NoGraphSimulation).
    self._master.expand_model_fields(fields)
    # Worker: must also carry these fields (recreates its own small graphs).
    self._worker.expand_model_fields(fields)
    # Both replaced model arrays under us; rebuild the cached copy views.
    self._rebuild_model_pairs()
    # Any already-captured fused graph holds pointers to the old model
    # arrays (same caveat Simulation.create_graph() documents for its own
    # graphs) -- drop it so the next step()/forward() call recaptures fresh.
    self._step_graph = None
    self._forward_graph = None

  def recompute_constants(self, level) -> None:
    # Cheap, collision-free constant recompute (mass/inertia-derived values).
    # Safe to run directly on the master at full size.
    self._master.recompute_constants(level)

  # -- Chunked physics operations. --

  def _chunk_slice(self, i: int) -> slice:
    return slice(i * self.chunk_size, (i + 1) * self.chunk_size)

  def _copy_in(self, chunk: slice) -> None:
    _apply_copy_pairs(self._model_in_pairs, chunk, slice(None))
    _apply_copy_pairs(self._data_in_pairs, chunk, slice(None))

  def _copy_out(self, chunk: slice) -> None:
    _apply_copy_pairs(self._data_out_pairs, slice(None), chunk)

  def step(self) -> None:
    if self._step_graph is None:
      self._step_graph = self._capture_fused(mjwarp.step)
    with wp.ScopedDevice(self._worker.wp_device):
      wp.capture_launch(self._step_graph)

  def forward(self) -> None:
    if self._forward_graph is None:
      self._forward_graph = self._capture_fused(mjwarp.forward)
    with wp.ScopedDevice(self._worker.wp_device):
      wp.capture_launch(self._forward_graph)

  def sense(self) -> None:
    # No sensor context supported yet; nothing to do (matches
    # Simulation.sense()'s own no-op when self._sensor_context is None).
    return

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      for i in range(self.num_chunks):
        chunk = self._chunk_slice(i)
        self._copy_in(chunk)
        self._worker.reset(None)
        self._copy_out(chunk)
      return

    if env_ids.numel() == 0:
      return

    chunk_of = torch.div(env_ids, self.chunk_size, rounding_mode="floor")
    local_ids = env_ids - chunk_of * self.chunk_size
    for i in range(self.num_chunks):
      mask = chunk_of == i
      if not bool(mask.any()):
        continue
      chunk = self._chunk_slice(i)
      self._copy_in(chunk)
      self._worker.reset(local_ids[mask])
      self._copy_out(chunk)


# mjlab version this module was built and validated against. A version
# mismatch does not block install (patch releases are usually fine) but is
# surfaced loudly, since this whole module depends on mjlab internals that
# are explicitly *not* part of its public API contract:
#   - Simulation.__init__'s exact positional/keyword signature
#   - Simulation holding exactly 4 CUDA-graph-backed ops: step/forward/sense/reset
#   - Simulation._should_use_cuda_graph() being overridable to force eager mode
#   - Simulation.expand_model_fields()/recompute_constants() existing as-is
#   - `mjlab.envs.manager_based_rl_env` importing Simulation by bare name
#     (`from mjlab.sim.sim import Simulation`) and calling it unqualified
# On a future mjlab bump, re-run the correctness/cross-contamination/memory
# checks under `scripts/` (or wherever this lands) before trusting this again.
_VALIDATED_MJLAB_VERSION = "1.6.0"
_REQUIRED_SIMULATION_ATTRS = (
  "step", "forward", "sense", "reset",
  "expand_model_fields", "recompute_constants", "_should_use_cuda_graph",
)


def _assert_mjlab_compatible(env_mod) -> None:
  try:
    installed = importlib.metadata.version("mjlab")
  except importlib.metadata.PackageNotFoundError:
    installed = "unknown"
  if installed != _VALIDATED_MJLAB_VERSION:
    warnings.warn(
      f"ChunkedSimulation was validated against mjlab=={_VALIDATED_MJLAB_VERSION}, "
      f"but mjlab=={installed} is installed. The CUDA-graph/memory-pool "
      "behavior this relies on is an internal implementation detail, not a "
      "public API contract -- re-run the correctness/memory checks before "
      "trusting this.",
      stacklevel=2,
    )

  if getattr(env_mod, "Simulation", None) is not Simulation:
    raise RuntimeError(
      "mjlab.envs.manager_based_rl_env.Simulation is not mjlab.sim.Simulation "
      "(or was already patched to something else) -- the module layout this "
      "monkeypatch depends on has changed upstream. Update "
      "install_chunked_simulation() to match before using it."
    )

  missing = [a for a in _REQUIRED_SIMULATION_ATTRS if not hasattr(Simulation, a)]
  if missing:
    raise RuntimeError(
      f"mjlab.sim.Simulation is missing expected method(s) {missing}; "
      "ChunkedSimulation's step/forward/sense/reset/expand_model_fields/"
      "recompute_constants/_should_use_cuda_graph overrides are no longer "
      "valid against this mjlab version. Update ChunkedSimulation to match."
    )


def install_chunked_simulation(chunk_size: int) -> None:
  """Monkeypatch ManagerBasedRlEnv to build a ChunkedSimulation, not a Simulation.

  ``mjlab.envs.manager_based_rl_env`` does ``from mjlab.sim.sim import
  Simulation`` then calls the bare name ``Simulation(...)`` in
  ``ManagerBasedRlEnv.__init__``; Python resolves that name from the module's
  globals at call time, so replacing it here redirects every subsequent env
  construction without touching mjlab's source. Idempotent re-install with a
  different ``chunk_size`` is fine (each call rebinds the same symbol).

  Raises loudly (rather than silently no-op'ing or miscompiling) if mjlab's
  internals have moved out from under this module -- see
  ``_assert_mjlab_compatible``.
  """
  import mjlab.envs.manager_based_rl_env as _env_mod

  _assert_mjlab_compatible(_env_mod)

  @functools.wraps(ChunkedSimulation)
  def _factory(num_envs, cfg, model=None, device="cuda:0", *, spec=None, variant_info=None):
    return ChunkedSimulation(
      num_envs=num_envs,
      cfg=cfg,
      chunk_size=chunk_size,
      model=model,
      device=device,
      spec=spec,
      variant_info=variant_info,
    )

  _env_mod.Simulation = _factory
  print(f"[INFO] ChunkedSimulation installed: chunk_size={chunk_size}")
