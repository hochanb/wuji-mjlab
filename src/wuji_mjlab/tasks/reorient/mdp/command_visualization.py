# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer.debug_visualizer import NullDebugVisualizer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

if TYPE_CHECKING:
  import mujoco
  import viser
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_GOAL_VIS_Z_OFFSET = 0.15
_STATUS_VIS_X_OFFSET = 0.06
_STATUS_VIS_RADIUS = 0.012
_FRAME_TRIAD_LEN = 0.10
_FRAME_TRIAD_RADIUS = 0.005
_TAG_TRIAD_COLORS = (
  (0.0, 0.85, 0.85),
  (0.85, 0.0, 0.85),
  (0.9, 0.85, 0.0),
)
_GOAL_RGBA = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
_GOAL_POS_RGBA = (1.0, 0.6, 0.0, 0.8)
_GOAL_POS_VIS_RADIUS = 0.006
_GOAL_PRIMITIVE_RGBA = (1.0, 0.5, 0.0, 0.35)
_GHOST_PART_ALPHA = 0.35
_CUBE_TRIAD_LEN = 0.045
_CUBE_TRIAD_RADIUS = 0.003
_GHOST_BOX_HALF = np.array([0.027, 0.027, 0.027], dtype=np.float64)
_WARNED_VISUALIZER_TYPES: set[type] = set()


def native_visualization_context(
  visualizer: "DebugVisualizer",
) -> tuple["mujoco.MjModel", "mujoco.MjvScene"] | None:
  if isinstance(visualizer, MujocoNativeDebugVisualizer):
    # mjlab 1.6 exposes no public scene/material API for textured debug boxes.
    return visualizer._ref_mj_model, visualizer._mjv_scene
  return None


def goal_vis_position_above_object(
  object_pos: np.ndarray,
  z_offset: float = _GOAL_VIS_Z_OFFSET,
) -> np.ndarray:
  vis_pos = np.array(object_pos, copy=True)
  vis_pos[2] += z_offset
  return vis_pos


def goal_vis_env_indices(num_envs: int) -> list[int]:
  return list(range(num_envs))


def reorient_status_color_rgba(success: bool) -> tuple[float, float, float, float]:
  if success:
    return (0.2, 0.9, 0.2, 0.95)
  return (0.95, 0.2, 0.2, 0.95)


def format_reorient_status_markdown(
  ori_error_rad: float, success: bool, pos_error_m: float | None = None
) -> str:
  ori_error_deg = float(np.degrees(ori_error_rad))
  pos_line = "" if pos_error_m is None else f"- pos_error: {pos_error_m * 100:.2f} cm\n"
  return (
    "### Reorient Policy Status\n"
    f"- ori_error: {ori_error_deg:.1f} deg\n"
    f"{pos_line}"
    f"- success: {success}\n"
  )


def update_reorient_status_markdown(
  status_markdown,
  get_env_idx: Callable[[], int],
  num_envs: int,
  policy_status_for_env: Callable[[int], tuple[float, bool]],
  pos_error_for_env: Callable[[int], float] | None = None,
) -> None:
  if num_envs <= 0:
    return
  env_idx = int(np.clip(get_env_idx(), 0, num_envs - 1))
  ori_error_rad, is_success = policy_status_for_env(env_idx)
  status_markdown.content = format_reorient_status_markdown(
    ori_error_rad=ori_error_rad,
    success=is_success,
    pos_error_m=None if pos_error_for_env is None else pos_error_for_env(env_idx),
  )


def draw_reorient_frame_triads(
  *,
  visualizer: "DebugVisualizer",
  num_envs: int,
  palm_pose_w: "torch.Tensor",
  tag_pose_w: "torch.Tensor",
) -> None:
  palm_pos_t = palm_pose_w[:, 0:3]
  palm_quat_t = palm_pose_w[:, 3:7]
  palm_R = matrix_from_quat(palm_quat_t).detach().cpu().numpy()
  palm_pos = palm_pos_t.detach().cpu().numpy()

  tag_pos_t = tag_pose_w[:, 0:3]
  tag_quat_t = tag_pose_w[:, 3:7]
  tag_R = matrix_from_quat(tag_quat_t).detach().cpu().numpy()
  tag_pos = tag_pos_t.detach().cpu().numpy()

  for env_idx in goal_vis_env_indices(num_envs):
    visualizer.add_frame(
      position=palm_pos[env_idx].astype(np.float32),
      rotation_matrix=palm_R[env_idx].astype(np.float32),
      scale=_FRAME_TRIAD_LEN,
      axis_radius=_FRAME_TRIAD_RADIUS,
      alpha=1.0,
      label=f"palm_frame_env{env_idx}",
    )
    visualizer.add_frame(
      position=tag_pos[env_idx].astype(np.float32),
      rotation_matrix=tag_R[env_idx].astype(np.float32),
      scale=_FRAME_TRIAD_LEN,
      axis_radius=_FRAME_TRIAD_RADIUS,
      alpha=1.0,
      axis_colors=_TAG_TRIAD_COLORS,
      label=f"tag_frame_env{env_idx}",
    )


@dataclass(frozen=True)
class GhostPart:
  """One visual primitive of the object, in the object's body frame."""

  kind: str  # "box" | "cylinder"
  size: np.ndarray  # box half extents | (radius, half length)
  pos: np.ndarray
  rot: np.ndarray  # 3x3
  rgba: tuple[float, float, float, float]


def object_ghost_parts(mj_model, geom_ids) -> list[GhostPart]:
  """Visual (group 2) box/cylinder geoms of an object, for a goal ghost.

  Textured geoms (e.g. the ArUco cube) get the plain goal tint; others keep
  their colour. Other geom types (meshes) are skipped.
  """
  import mujoco

  kinds = {
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
  }
  parts = []
  for gid in (int(g) for g in geom_ids):
    kind = kinds.get(int(mj_model.geom_type[gid]))
    if kind is None or mj_model.geom_group[gid] != 2:
      continue
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, mj_model.geom_quat[gid])
    if mj_model.geom_matid[gid] >= 0:
      rgba = _GOAL_PRIMITIVE_RGBA
    else:
      r, g, b, _ = (float(c) for c in mj_model.geom_rgba[gid])
      rgba = (r, g, b, _GHOST_PART_ALPHA)
    size = mj_model.geom_size[gid].copy()
    parts.append(
      GhostPart(
        kind=kind,
        size=size if kind == "box" else size[:2],
        pos=mj_model.geom_pos[gid].copy(),
        rot=rot.reshape(3, 3),
        rgba=rgba,
      )
    )
  return parts


@dataclass
class ReorientCommandVisualization:
  entity_name: str
  ghost_mat_id: int = -1
  ghost_box_size: np.ndarray = field(default_factory=lambda: _GHOST_BOX_HALF.copy())
  _ghost_resolved: bool = False
  status_markdown: object | None = None
  get_env_idx: Callable[[], int] | None = None
  # Set when the command carries a position goal; adds pos_error to the status.
  pos_error_for_env: Callable[[int], float] | None = None
  # The object's own primitives; when set, viewers without MjvScene access draw
  # the goal as a translucent copy of the object instead of a plain box.
  ghost_parts: list[GhostPart] | None = None

  def create_gui(
    self,
    *,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    with server.gui.add_folder(name.capitalize()):
      self.status_markdown = server.gui.add_markdown("")
    self.get_env_idx = get_env_idx

  def attach_status_targets(
    self,
    *,
    status_markdown,
    get_env_idx: Callable[[], int],
  ) -> None:
    self.status_markdown = status_markdown
    self.get_env_idx = get_env_idx

  def update_status_gui(
    self,
    *,
    num_envs: int,
    policy_status_for_env: Callable[[int], tuple[float, bool]],
  ) -> None:
    if self.status_markdown is None or self.get_env_idx is None:
      return
    update_reorient_status_markdown(
      status_markdown=self.status_markdown,
      get_env_idx=self.get_env_idx,
      num_envs=num_envs,
      policy_status_for_env=policy_status_for_env,
      pos_error_for_env=self.pos_error_for_env,
    )

  def _ensure_ghost_material(self, mj_model) -> bool:
    if self._ghost_resolved:
      return self.ghost_mat_id >= 0
    self._ghost_resolved = True

    import mujoco

    cube_body = mujoco.mj_name2id(
      mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{self.entity_name}/cube"
    )

    for geom_idx in range(mj_model.ngeom):
      if (
        mj_model.geom_bodyid[geom_idx] == cube_body
        and mj_model.geom_group[geom_idx] == 2
      ):
        self.ghost_mat_id = int(mj_model.geom_matid[geom_idx])
        self.ghost_box_size = mj_model.geom_size[geom_idx].copy()
        if self.ghost_mat_id >= 0:
          return True
        break
    warnings.warn(
      f"No visual material found for {self.entity_name}/cube; "
      "the textured reorient goal cannot be drawn.",
      RuntimeWarning,
      stacklevel=2,
    )
    return False

  def draw_debug_visuals(
    self,
    *,
    visualizer: "DebugVisualizer",
    num_envs: int,
    palm_pose_w: "torch.Tensor",
    tag_pose_w: "torch.Tensor",
    object_pos_w: "torch.Tensor",
    goal_quat_w: "torch.Tensor",
    policy_status_for_env: Callable[[int], tuple[float, bool]],
    goal_pos_w: "torch.Tensor | None" = None,
    object_quat_w: "torch.Tensor | None" = None,
  ) -> None:
    """Draw the goal ghost.

    With a position goal the ghost sits at the goal pose. Without one, the
    translucent primitive ghost (viser etc.) overlaps the live object so the
    orientation mismatch reads directly; the textured native ghost floats
    above the live object instead, since an overlapping textured box would
    z-fight with the cube once aligned.
    """
    draw_reorient_frame_triads(
      visualizer=visualizer,
      num_envs=num_envs,
      palm_pose_w=palm_pose_w,
      tag_pose_w=tag_pose_w,
    )

    def ghost_pos(env_idx: int) -> np.ndarray:
      if goal_pos_w is not None:
        return goal_pos_w[env_idx].detach().cpu().numpy()
      return goal_vis_position_above_object(
        object_pos_w[env_idx].detach().cpu().numpy()
      )

    def overlay_pos(env_idx: int) -> np.ndarray:
      if goal_pos_w is not None:
        return goal_pos_w[env_idx].detach().cpu().numpy()
      return object_pos_w[env_idx].detach().cpu().numpy()

    context = native_visualization_context(visualizer)
    if context is None:
      if not isinstance(visualizer, NullDebugVisualizer):
        self._draw_primitive_goal(
          visualizer=visualizer,
          num_envs=num_envs,
          ghost_pos=overlay_pos,
          object_pos_w=object_pos_w,
          object_quat_w=object_quat_w,
          goal_quat_w=goal_quat_w,
          goal_pos_w=goal_pos_w,
          policy_status_for_env=policy_status_for_env,
        )
      return

    mj_model, mj_scene = context
    if not self._ensure_ghost_material(mj_model):
      return

    import mujoco

    goal_rot = matrix_from_quat(goal_quat_w)

    for env_idx in goal_vis_env_indices(num_envs):
      if mj_scene.ngeom >= mj_scene.maxgeom:
        break

      ori_error_rad, is_success = policy_status_for_env(env_idx)
      vis_pos = ghost_pos(env_idx)
      geom = mj_scene.geoms[mj_scene.ngeom]
      mj_scene.ngeom += 1
      geom.category = mujoco.mjtCatBit.mjCAT_DECOR
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX.value,
        self.ghost_box_size,
        vis_pos.astype(np.float64),
        goal_rot[env_idx].detach().cpu().numpy().reshape(-1).astype(np.float64),
        _GOAL_RGBA,
      )
      # Same arucocube material as the live cube; the cube texture samples by
      # face direction on a box, so no mesh dataid / texcoords are needed.
      geom.matid = self.ghost_mat_id

      status_pos = vis_pos.copy()
      status_pos[0] += _STATUS_VIS_X_OFFSET
      visualizer.add_sphere(
        center=status_pos,
        radius=_STATUS_VIS_RADIUS,
        color=reorient_status_color_rgba(is_success),
        label=(
          f"reorient_status_env{env_idx}"
          f"_err_{np.degrees(ori_error_rad):.1f}deg"
          f"_success_{is_success}"
        ),
      )

  def _draw_primitive_goal(
    self,
    *,
    visualizer: "DebugVisualizer",
    num_envs: int,
    ghost_pos: Callable[[int], np.ndarray],
    object_pos_w: "torch.Tensor",
    object_quat_w: "torch.Tensor | None",
    goal_quat_w: "torch.Tensor",
    goal_pos_w: "torch.Tensor | None",
    policy_status_for_env: Callable[[int], tuple[float, bool]],
  ) -> None:
    """Untextured goal for viewers without MjvScene access (e.g. viser).

    A plain box is rotation-ambiguous, so matching axis triads are drawn on
    the goal ghost and on the live cube: aligned triads mean aligned cubes.
    """
    goal_rot = matrix_from_quat(goal_quat_w).detach().cpu().numpy()
    cube_rot = (
      None
      if object_quat_w is None
      else matrix_from_quat(object_quat_w).detach().cpu().numpy()
    )
    # Every env, like the frame triads and the native ghost (not only the env
    # selected in the viewer GUI).
    for env_idx in goal_vis_env_indices(num_envs):
      ori_error_rad, is_success = policy_status_for_env(env_idx)
      vis_pos = ghost_pos(env_idx)
      if self.ghost_parts:
        self._draw_ghost_parts(visualizer, env_idx, vis_pos, goal_rot[env_idx])
      else:
        visualizer.add_box(
          center=vis_pos,
          size=self.ghost_box_size,
          mat=goal_rot[env_idx],
          color=_GOAL_PRIMITIVE_RGBA,
          label=f"reorient_goal_env{env_idx}",
        )
      visualizer.add_frame(
        position=vis_pos,
        rotation_matrix=goal_rot[env_idx],
        scale=_CUBE_TRIAD_LEN,
        axis_radius=_CUBE_TRIAD_RADIUS,
        alpha=0.6,
        label=f"reorient_goal_frame_env{env_idx}",
      )
      if cube_rot is not None:
        visualizer.add_frame(
          position=object_pos_w[env_idx].detach().cpu().numpy(),
          rotation_matrix=cube_rot[env_idx],
          scale=_CUBE_TRIAD_LEN,
          axis_radius=_CUBE_TRIAD_RADIUS,
          alpha=1.0,
          label=f"reorient_cube_frame_env{env_idx}",
        )
      if goal_pos_w is not None:
        visualizer.add_sphere(
          center=vis_pos,
          radius=_GOAL_POS_VIS_RADIUS,
          color=_GOAL_POS_RGBA,
          label=f"reorient_goal_pos_env{env_idx}",
        )
      status_pos = vis_pos.copy()
      status_pos[0] += _STATUS_VIS_X_OFFSET
      visualizer.add_sphere(
        center=status_pos,
        radius=_STATUS_VIS_RADIUS,
        color=reorient_status_color_rgba(is_success),
        label=(
          f"reorient_status_env{env_idx}"
          f"_err_{np.degrees(ori_error_rad):.1f}deg"
          f"_success_{is_success}"
        ),
      )

  def _draw_ghost_parts(
    self,
    visualizer: "DebugVisualizer",
    env_idx: int,
    origin: np.ndarray,
    rot: np.ndarray,
  ) -> None:
    """Translucent copy of the object's primitives at the goal pose."""
    for i, part in enumerate(self.ghost_parts or ()):
      center = origin + rot @ part.pos
      mat = rot @ part.rot
      label = f"reorient_goal_env{env_idx}_part{i}"
      if part.kind == "box":
        visualizer.add_box(
          center=center, size=part.size, mat=mat, color=part.rgba, label=label
        )
      else:
        radius, half_len = part.size
        axis = mat[:, 2] * half_len
        visualizer.add_cylinder(
          start=center - axis,
          end=center + axis,
          radius=radius,
          color=part.rgba,
          label=label,
        )
