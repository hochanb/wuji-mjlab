# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Replay small tensor functions of the env step as CUDA graphs.

An env step launches ~1,800 small kernels from Python; the GPU work is a
few ms, the launching tens of ms. A function that only reads the simulation
state and persistent buffers (updated in place) and computes tensors can be
captured once and replayed: one launch instead of a hundred.

Conditions for a function wrapped here: no CPU-GPU synchronisation, no Python
state changed per call (counters, branches on tensor values, step counts),
and every input tensor keeps its storage (in-place updates only). Caches are
bypassed while capturing (``state_cache.bypass``) so the graph holds the
whole computation. Outputs are the graph's static tensors: valid until the
next replay.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from wuji_mjlab.tasks.reorient.mdp import state_cache

ENABLED = os.environ.get("WUJI_GRAPHED_TERMS", "1") != "0"
CHECK = os.environ.get("WUJI_GRAPH_CHECK", "0") == "1"
"""``WUJI_GRAPH_CHECK=1``: after every replay, recompute eagerly and raise on
any difference (slow; for validating a wrapped function)."""


def _leaves(out):
  if isinstance(out, torch.Tensor):
    return [out]
  if isinstance(out, dict):
    return [v for k in sorted(out) for v in _leaves(out[k])]
  if isinstance(out, (tuple, list)):
    return [v for x in out for v in _leaves(x)]
  return []


class GraphedFn:
  """``fn()`` eagerly for ``warmup`` calls, then captured and replayed."""

  def __init__(
    self, fn: Callable[[], torch.Tensor], warmup: int = 3, name: str = "", check=True
  ):
    self.fn, self.warmup, self.name = fn, warmup, name or getattr(fn, "__name__", "")
    self.check = check and CHECK
    self._calls = 0
    self._graph: torch.cuda.CUDAGraph | None = None
    self._out = None

  def __call__(self):
    if self._graph is not None:
      self._graph.replay()
      if self.check:
        self._check()
      return self._out
    self._calls += 1
    if self._calls <= self.warmup or not torch.cuda.is_available():
      with state_cache.bypass():
        return self.fn()
    # Warm up on a side stream (allocator), then capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), state_cache.bypass():
      self.fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), state_cache.bypass():
      self._out = self.fn()
    self._graph = graph
    graph.replay()
    return self._out

  def _check(self) -> None:
    with state_cache.bypass():
      ref = self.fn()
    for got, want in zip(_leaves(self._out), _leaves(ref), strict=True):
      if not torch.equal(got, want):
        diff = (got.double() - want.double()).abs().max().item()
        raise AssertionError(f"graphed {self.name}: replay differs (max {diff:.3e})")


def graphed_term(func: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
  """A manager term ``func(env, **params)`` replayed as a CUDA graph per env
  and parameter set (eager on the CPU)."""

  def term(env, **params):
    if not ENABLED or "cuda" not in str(getattr(env, "device", "cpu")):
      return func(env, **params)
    graphs = env.__dict__.setdefault("_wuji_graphs", {})
    key = (func, tuple(sorted((k, repr(v)) for k, v in params.items())))
    if key not in graphs:
      # Random terms (sensor noise, outlier injection) cannot be checked
      # against a recompute: their draws differ.
      random = params.get("noise") is not None or params.get("injection_prob", 0) > 0
      graphs[key] = GraphedFn(
        lambda: func(env, **params), name=func.__name__, check=not random
      )
    return graphs[key]()

  term.__name__ = func.__name__
  term.__doc__ = func.__doc__
  term.__wrapped__ = func
  term.graphed = True
  return term


def graph_plain_terms(terms, eager=()) -> None:
  """Wrap every plain-function term of ``terms`` (name -> term cfg) with
  ``graphed_term``, except class terms, already wrapped ones and ``eager``."""
  for name, term in terms.items():
    func = term.func
    if name in eager or isinstance(func, type) or getattr(func, "graphed", False):
      continue
    term.func = graphed_term(func)
