# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import os

from mjlab.utils.lab_api.tasks.importer import import_packages

# Opt-in memory-bounded physics: every env built after this module is
# imported runs sim.step()/forward()/reset() in fixed-size GPU chunks instead
# of all num_envs at once (see wuji_mjlab/rl/chunked_sim.py). Every training/
# eval entry point does `import wuji_mjlab.tasks` before constructing an env,
# so this one check covers all of them.
_chunk_size = os.environ.get("WUJI_SIM_CHUNK_SIZE")
if _chunk_size:
  from wuji_mjlab.rl.chunked_sim import install_chunked_simulation

  install_chunked_simulation(int(_chunk_size))

# mjlab's reset paths synchronise CPU and GPU ~100 times per reset (scalar
# row writes, .item() logging); swap in equivalent sync-free versions
# (wuji_mjlab/rl/sync_free_resets.py). WUJI_SYNC_FREE_RESETS=0 keeps mjlab's.
if os.environ.get("WUJI_SYNC_FREE_RESETS", "1") != "0":
  from wuji_mjlab.rl.sync_free_resets import install as _install_sync_free_resets

  _install_sync_free_resets()

_BLACKLIST_PKGS = [
  "common",
  ".mdp",
  ".rl",
]

import_packages(__name__, _BLACKLIST_PKGS)
