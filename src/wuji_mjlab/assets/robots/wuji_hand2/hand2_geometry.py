# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand 2 geometry used across tasks and tools (grasp synthesis, grasp
targets): hand dimensions only, no task settings."""

# Fingertip reference points (thumb first), as offsets in each
# right_finger<i>_tip site frame (+z proximal, +y towards the fingerprint; the
# site sits at the very end of the pad). Each is 30% of the pad length back from
# the end and HAND2_TIP_PAD_DEPTH in from the fingerprint surface.
HAND2_TIP_PAD_POINTS = (
  (0.0, 0.0031, 0.0072),
  (0.0, 0.0008, 0.0052),
  (0.0, 0.0008, 0.0052),
  (0.0, 0.0008, 0.0052),
  (0.0, 0.0008, 0.0052),
)
# Depth of HAND2_TIP_PAD_POINTS inside the pad [m].
HAND2_TIP_PAD_DEPTH = 0.005
