# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import shutil

import pytest
import torch
import yaml
from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
  GraspStageError,
  load_training,
)


def _example_root():
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
    EXAMPLE_TRAINING,
  )

  return EXAMPLE_TRAINING.parents[1]


@pytest.fixture
def assets(tmp_path):
  """A writable copy of the bundled grasp-stage assets."""
  root = tmp_path / "grasp_stage"
  shutil.copytree(_example_root(), root)
  return root


def _only(grasps):
  """The single grasp of a set, as a namespace of rows."""
  from types import SimpleNamespace

  assert len(grasps) == 1
  return SimpleNamespace(
    id=grasps.ids[0], **{a: getattr(grasps, a)[0].tolist() for a in grasps.ARRAYS}
  )


def _edit(path, fn):
  data = yaml.safe_load(path.read_text())
  fn(data)
  path.write_text(yaml.safe_dump(data))


def test_example_training_loads():
  training = load_training(_example_root() / "trainings" / "stapler_single.yaml")
  (stage,) = training.stages
  assert [o.name for o in stage.objects] == ["stapler"]
  assert len(stage.init_grasps["stapler"]) == 1
  goal = _only(stage.goal_grasps["stapler"])
  assert goal.tip_weights == [1.0] * 5 and goal.joint_weights == [0.0] * 5
  # Pose weights default to 1 when a grasp leaves them out (the init grasp).
  assert _only(stage.init_grasps["stapler"]).pose_weights == [1.0, 1.0]
  assert goal.pose_weights == [1.0, 1.0]
  assert training.total_iterations == 1000


@pytest.mark.parametrize(
  ("target", "edit", "message"),
  [
    ("goal", lambda d: d["grasps"][0]["tip"].update(weights=[1, 1, 1, 1, 1.5]), "[0, 1]"),
    ("goal", lambda d: d["grasps"][0]["joint"].update(q=[0.0] * 19), "20 finite"),
    ("goal", lambda d: d["grasps"][0]["object_pose"].update(quat_wxyz=[1, 1, 0, 0]), "unit"),
    ("goal", lambda d: d.update(hand="other_hand"), "hand"),
    ("goal", lambda d: d["grasps"][0]["object_pose"].update(weights={"pos": 2, "rot": 1}), "[0, 1]"),
    ("goal", lambda d: d["grasps"][0]["object_pose"].update(weights={"pos": 1}), "pos and rot"),
    ("goal", lambda d: d.update(frames={"object_pose": "world", "tip": "object"}), "frames"),
    ("goal", lambda d: d["grasps"].append(d["grasps"][0]), "duplicate id"),
    ("training", lambda d: d["stages"][0].update(objects=["mug"]), "not found"),
    ("training", lambda d: d["stages"][0]["goal_grasps"].update(stapler=["grasps/none"]), "not found"),
    ("training", lambda d: d["stages"][0].update(objects=["stapler", "stapler2"]), "several objects"),
    ("training", lambda d: d["stages"][0].update(advance={"after_iterations": 0}), "positive"),
    ("training", lambda d: d["stages"].append(dict(d["stages"][0])), "duplicate stage"),
    ("training", lambda d: d.update(encoding={"name": "bps", "dim": 8}), "encoding"),
  ],
)  # fmt: skip
def test_validation_rejects(assets, target, edit, message):
  training = assets / "trainings" / "stapler_single.yaml"
  path = {
    "goal": assets / "objects" / "stapler" / "grasps" / "goal_grasp" / "grasps.yaml",
    "training": training,
  }[target]
  _edit(path, edit)
  with pytest.raises(GraspStageError, match=message.replace("[", r"\[")):
    load_training(training)


def test_two_stage_training_merges_grasp_sets(assets):
  folder = assets / "objects" / "stapler" / "grasps"
  shutil.copytree(folder / "goal_grasp", folder / "goal_b")
  _edit(folder / "goal_b" / "grasps.yaml", lambda d: d["grasps"][0].update(id="b"))
  training = assets / "trainings" / "stapler_single.yaml"

  def two_stages(d):
    second = dict(d["stages"][0], name="s1")
    second["goal_grasps"] = {"stapler": ["grasps/goal_grasp", "grasps/goal_b"]}
    d["stages"].append(second)

  _edit(training, two_stages)
  spec = load_training(training)
  assert [s.name for s in spec.stages] == ["s0", "s1"]
  assert spec.stages[1].goal_grasps["stapler"].ids == ("side_grasp", "b")
  assert spec.total_iterations == 2000


@pytest.fixture(scope="module")
def stage_env():
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
    example_stage_env_cfg,
  )

  cfg = example_stage_env_cfg(play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield env
  env.close()


def _in_palm(cmd, pos_w, quat_w):
  from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul

  palm = cmd.robot.data.body_link_pose_w[:, cmd.palm_body_id]
  return (
    quat_apply_inverse(palm[:, 3:7], pos_w - palm[:, :3]),
    quat_mul(quat_inv(palm[:, 3:7]), quat_w),
  )


def test_stage_env_resets_into_the_init_grasp_with_the_goal_grasp(stage_env):
  from mjlab.utils.lab_api.math import quat_error_magnitude

  training = load_training(_example_root() / "trainings" / "stapler_single.yaml")
  init = _only(training.stages[0].init_grasps["stapler"])
  goal = _only(training.stages[0].goal_grasps["stapler"])
  stage_env.reset()
  stage_env.sim.forward()
  cmd = stage_env.command_manager.get_term("reorient_command")
  robot, obj = cmd.robot, cmd.object
  ids, _ = robot.find_joints(".*_finger.*_joint.*")
  q = torch.tensor(init.joint_q).expand(2, -1)
  torch.testing.assert_close(robot.data.joint_pos[:, ids], q, atol=1e-5, rtol=0)
  pos, quat = _in_palm(cmd, obj.data.root_link_pos_w, obj.data.root_link_quat_w)
  torch.testing.assert_close(
    pos, torch.tensor(init.object_pos).expand(2, 3), atol=1e-4, rtol=0
  )
  assert torch.all(
    quat_error_magnitude(quat, torch.tensor(init.object_quat).expand(2, 4)) < 1e-3
  )
  gpos, gquat = _in_palm(cmd, cmd.goal_pos_w, cmd.goal_quat_w)
  torch.testing.assert_close(
    gpos, torch.tensor(goal.object_pos).expand(2, 3), atol=1e-4, rtol=0
  )
  assert torch.all(
    quat_error_magnitude(gquat, torch.tensor(goal.object_quat).expand(2, 4)) < 1e-3
  )
  torch.testing.assert_close(
    cmd.tip_weights, torch.tensor(goal.tip_weights).expand(2, 5)
  )


def test_grasp_observations_and_success_gate(stage_env):
  from wuji_mjlab.tasks.reorient.grasp_stage import mdp as gmdp

  stage_env.reset()
  stage_env.sim.forward()
  cmd = stage_env.command_manager.get_term("reorient_command")
  for group in ("policy", "critic"):
    names = stage_env.observation_manager.active_terms[group]
    assert {"grasp_tip_error", "grasp_weights", "grasp_joint_goal"} <= set(names)
  weights = gmdp.grasp_weights(stage_env)
  assert weights.shape == (2, 12)
  err = gmdp.grasp_tip_error(stage_env).view(2, 5, 3)
  # Fingers open, stapler resting: every active tip is centimetres off.
  assert torch.all(err.norm(dim=-1) > 0.01)
  assert not cmd._success_gate().any()
  saved = cmd.tip_weights.clone()
  cmd.tip_weights[:] = 0.0  # no tip or joint condition: the gate passes
  assert cmd._success_gate().all()
  assert torch.all(gmdp.grasp_tip_error(stage_env) == 0.0)  # masked out
  cmd.tip_weights[:] = saved


def test_grasp_errors_are_computed_once_per_sim_state(stage_env, monkeypatch):
  from wuji_mjlab.tasks.reorient.grasp_stage import command as command_module

  stage_env.reset()
  cmd = stage_env.command_manager.get_term("reorient_command")
  zero = torch.zeros(2, stage_env.action_manager.total_action_dim)
  stage_env.step(zero)
  calls = []
  original = command_module.quat_apply_inverse

  def counting(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)

  monkeypatch.setattr(command_module, "quat_apply_inverse", counting)
  for _ in range(3):
    stage_env.step(zero)
  # Rewards (after physics) and command + observations (after forward).
  assert len(calls) == 2 * 3
  # The cached state-B values match a fresh computation of the same state.
  cached = (
    cmd.tip_offset_b().clone(),
    cmd.tip_cos().clone(),
    cmd.joint_error().clone(),
  )
  cmd.invalidate_grasp_errors()
  fresh = (cmd.tip_offset_b(), cmd.tip_cos(), cmd.joint_error())
  for a, b in zip(cached, fresh, strict=True):
    torch.testing.assert_close(a, b)


def test_pose_weights_switch_pose_parts_off(stage_env):
  from wuji_mjlab.tasks.reorient.grasp_stage import mdp as gmdp
  from wuji_mjlab.tasks.reorient.mdp.rewards import orientation_alignment

  stage_env.reset()
  cmd = stage_env.command_manager.get_term("reorient_command")
  zero = torch.zeros(2, stage_env.action_manager.total_action_dim)
  stage_env.step(zero)
  # Resting on the open hand, far from the goal pose: not within threshold.
  assert not cmd.within_threshold.any()
  rot = gmdp.grasp_rot_err_6d(stage_env)
  assert torch.all(rot.abs().sum(-1) > 0)
  full = orientation_alignment(stage_env, "reorient_command")
  torch.testing.assert_close(gmdp.grasp_orientation_alignment(stage_env), full)

  cmd.pose_weights[0] = torch.tensor([0.0, 0.0])  # env 0: pose not required
  cmd.pose_weights[1] = torch.tensor([0.0, 0.5])  # env 1: orientation at half weight
  saved = cmd.tip_weights.clone()
  cmd.tip_weights[:] = 0.0
  stage_env.step(zero)
  torch.testing.assert_close(cmd.within_threshold, torch.tensor([True, False]))
  assert torch.all(gmdp.grasp_rot_err_6d(stage_env)[0] == 0.0)
  assert torch.all(gmdp.grasp_pos_err(stage_env) == 0.0)
  scaled = gmdp.grasp_orientation_alignment(stage_env)
  full = orientation_alignment(stage_env, "reorient_command")
  torch.testing.assert_close(scaled, torch.stack((0.0 * full[0], 0.5 * full[1])))
  cmd.tip_weights[:] = saved
  cmd.pose_weights[:] = 1.0


def test_observations_are_env_then_condition_and_masks_are_exact():
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
    example_stage_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.grasp_stage import mdp as gmdp

  cfg = example_stage_env_cfg()  # training cfg: policy noise on
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    layout = gmdp.observation_layout(env)
    for group in ("policy", "critic"):
      rows = layout[group]
      kinds = [r["kind"] for r in rows]
      n_env = kinds.count("env")
      assert kinds == ["env"] * n_env + ["condition"] * (len(kinds) - n_env)
      assert all(r["noise"] is None for r in rows if r["kind"] == "condition")
      assert all(
        a["slice"][1] == b["slice"][0] for a, b in zip(rows, rows[1:], strict=False)
      )
    policy = {r["term"]: r for r in layout["policy"]}
    assert policy["cube_ori_error"]["noise"] == "GaussianNoiseCfg"  # still noised
    assert policy["grasp_tip_error"]["noise"] == "UniformNoiseCfg"
    cmd = env.command_manager.get_term("reorient_command")
    lo, hi = policy["cube_ori_error"]["slice"]
    a = env.observation_manager.compute(update_history=True)["policy"][:, lo:hi]
    b = env.observation_manager.compute(update_history=True)["policy"][:, lo:hi]
    assert not torch.equal(a, b)  # required parts keep their sensor noise
    lo, hi = policy["grasp_tip_error"]["slice"]
    a = env.observation_manager.compute(update_history=True)["policy"][:, lo:hi]
    b = env.observation_manager.compute(update_history=True)["policy"][:, lo:hi]
    assert not torch.equal(a, b) and (a - b).abs().max() <= 0.016 + 1e-6
    cmd.pose_weights[:] = torch.tensor([0.0, 0.0])
    cmd.tip_weights[:, 1:] = 0.0  # only the thumb stays required
    for _ in range(3):  # fill the 3-step history with masked readings
      obs = env.observation_manager.compute(update_history=True)["policy"]
    for name in ("cube_ori_error", "goal_pos_err"):
      lo, hi = policy[name]["slice"]
      assert torch.all(obs[:, lo:hi] == 0.0)  # exactly 0, no noise
    lo, hi = policy["grasp_tip_error"]["slice"]
    tips = obs[:, lo:hi].view(2, 3, 5, 3)  # history x finger x xyz
    assert torch.all(tips[:, :, 1:] == 0.0) and torch.all(tips[:, :, 0] != 0.0)
    lo, hi = policy["grasp_weights"]["slice"]
    torch.testing.assert_close(obs[:, lo : lo + 2], torch.zeros(2, 2))
  finally:
    env.close()


def test_npz_grasp_sets_round_trip_and_validate(assets):
  import numpy as np
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
    load_grasp_set,
    save_grasp_set_npz,
  )

  training = assets / "trainings" / "stapler_single.yaml"
  folder = assets / "objects" / "stapler" / "grasps" / "goal_grasp"
  yml = load_grasp_set(folder, "wuji_hand2_right")
  save_grasp_set_npz(folder, yml, "wuji_hand2_right")
  with pytest.raises(GraspStageError, match="both grasps.yaml and grasps.npz"):
    load_training(training)
  (folder / "grasps.yaml").unlink()
  npz = load_training(training).stages[0].goal_grasps["stapler"]
  assert npz.ids == yml.ids
  for a in yml.ARRAYS:
    np.testing.assert_allclose(getattr(npz, a), getattr(yml, a), atol=1e-6)
  bad = npz.tip_weights.copy()
  bad[0, 2] = 1.5
  save_grasp_set_npz(folder, type(npz)(npz.ids, *(
    bad if a == "tip_weights" else getattr(npz, a) for a in npz.ARRAYS
  )), "wuji_hand2_right")  # fmt: skip
  with pytest.raises(GraspStageError, match=r"tip_weights of grasp 'side_grasp'"):
    load_training(training)


def test_ten_thousand_grasps_load_fast(tmp_path):
  import time

  import numpy as np
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import (
    GraspSet,
    load_grasp_set,
    save_grasp_set_npz,
  )

  g, rng = 10_000, np.random.default_rng(0)
  unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
  grasps = GraspSet(
    tuple(f"g{i}" for i in range(g)),
    rng.normal(size=(g, 3)),
    unit(rng.normal(size=(g, 4))),
    np.ones((g, 2)),
    rng.normal(size=(g, 5, 3)),
    unit(rng.normal(size=(g, 5, 3))),
    np.ones((g, 5)),
    rng.uniform(-1, 1, size=(g, 20)),
    np.ones((g, 5)),
  )
  save_grasp_set_npz(tmp_path, grasps, "wuji_hand2_right")
  start = time.perf_counter()
  loaded = load_grasp_set(tmp_path, "wuji_hand2_right")
  assert len(loaded) == g
  assert time.perf_counter() - start < 2.0


def test_reset_event_builds_its_tensors_once(stage_env):
  stage_env.reset()
  cache = stage_env._grasp_stage_cache
  tensors = {k: v[1] for k, v in cache.items() if k[0] == "tensor"}
  assert len(tensors) == 3  # object pos, quat and joint goals of the init set
  stage_env.reset()
  for k, t in tensors.items():
    assert stage_env._grasp_stage_cache[k][1] is t


def test_env_cfg_dump_summarises_grasp_data():
  from wuji_mjlab.tasks.reorient.config.wuji_hand2.grasp_stage import (
    env_cfg_for_dump,
    example_stage_env_cfg,
  )

  data = env_cfg_for_dump(example_stage_env_cfg())
  command = data["commands"]["reorient_command"]
  assert command["goal_tip_points"] == "<1 entries, see grasp files>"
  params = data["events"]["reset_object_pose"]["params"]
  assert params["joint_q"] == "<1 entries, see grasp files>"


def _set_q(path, q, weights=None):
  def fn(d):
    d["grasps"][0]["joint"]["q"] = q
    if weights is not None:
      d["grasps"][0]["joint"]["weights"] = weights

  _edit(path, fn)


def test_joint_angles_are_checked_against_the_hand_limits(assets):
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import hand_joint_limits

  training = assets / "trainings" / "stapler_single.yaml"
  grasps = assets / "objects" / "stapler" / "grasps"
  _, hard, soft = hand_joint_limits("wuji_hand2_right")
  # Initial grasp beyond a joint's range: the hand cannot be put there.
  q = [0.0] * 20
  q[1] = float(hard[1, 1]) + 0.01  # right_finger1_joint2
  _set_q(grasps / "init_rest" / "grasps.yaml", q)
  with pytest.raises(
    GraspStageError, match=r"right_finger1_joint2 .* outside its range"
  ):
    load_training(training)
  _set_q(grasps / "init_rest" / "grasps.yaml", [0.0] * 20)
  # Goal between the soft limit and the range: fine while the finger's joint
  # goal is off, rejected once it is weighted.
  q = [0.0] * 20
  q[5] = float(soft[5, 1] + hard[5, 1]) / 2  # right_finger2_joint2
  _set_q(grasps / "goal_grasp" / "grasps.yaml", q, [0, 0, 0, 0, 0])
  load_training(training)
  _set_q(grasps / "goal_grasp" / "grasps.yaml", q, [0, 1, 0, 0, 0])
  with pytest.raises(GraspStageError, match=r"right_finger2_joint2 .* soft limits"):
    load_training(training)


def test_joint_limits_match_the_env(stage_env):
  import numpy as np
  from wuji_mjlab.tasks.reorient.grasp_stage.schema import hand_joint_limits

  names, _, soft = hand_joint_limits("wuji_hand2_right")
  robot = stage_env.scene["robot"]
  ids, found = robot.find_joints(list(names), preserve_order=True)
  assert list(found) == list(names)
  env_soft = robot.data.soft_joint_pos_limits[0, ids].cpu().numpy()
  np.testing.assert_allclose(env_soft, soft, atol=1e-5)
