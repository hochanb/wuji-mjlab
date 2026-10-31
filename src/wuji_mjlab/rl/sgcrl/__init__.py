# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Single-goal contrastive RL (arXiv:2408.05804)."""

from .config import SgcrlCfg
from .trainer import SgcrlTrainer

__all__ = ["SgcrlCfg", "SgcrlTrainer"]
