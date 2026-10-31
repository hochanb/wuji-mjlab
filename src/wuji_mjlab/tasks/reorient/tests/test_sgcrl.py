# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import math

import pytest
import torch
from wuji_mjlab.rl.sgcrl.buffer import TrajectoryBuffer
from wuji_mjlab.rl.sgcrl.networks import TanhGaussianActor

_TASK = "WujiHand2_ReorientStaplerFixedGoalWrist_SGCRL_50Hz"


def _filled_buffer(steps: int, episode_len: int, num_envs: int = 3, capacity=64):
  """Each stored state and goal is its global step; env e ends an episode
  every ``episode_len + e`` steps."""
  buf = TrajectoryBuffer(capacity, num_envs, 1, 1, 1, "cpu")
  for t in range(steps):
    x = torch.full((num_envs, 1), float(t))
    lengths = torch.arange(num_envs) + episode_len
    done = torch.tensor([(t + 1) % int(n) == 0 for n in lengths])
    buf.add(x, x, x, done)
  return buf, torch.arange(num_envs) + episode_len


@pytest.mark.parametrize("absorb", [True, False])
def test_future_goal_is_later_in_the_same_episode(absorb):
  buf, lengths = _filled_buffer(steps=150, episode_len=7)
  s, _, g = buf.sample(4096, discount=0.9, absorb=absorb)
  # States and goals record their global step; recover each sample's env from
  # the episode boundaries: goal and state lie in the same episode for some env.
  k = g - s
  # Absorbing: an episode's last state is its own future.
  assert torch.all(k >= (0 if absorb else 1))
  same_episode = torch.zeros_like(k, dtype=torch.bool)
  for n in lengths.tolist():
    same_episode |= torch.div(s, n, rounding_mode="floor") == torch.div(
      g, n, rounding_mode="floor"
    )
  assert torch.all(same_episode)
  assert torch.all(g <= 149)  # never beyond the newest step
  assert torch.all(s >= 150 - 64)  # only slots still in the ring


def test_ended_episode_absorbs_into_its_last_state():
  # One env, episodes of 5 steps (0-4, 5-9, ...): from the first state of an
  # episode, offsets past its last state land on that state with probability
  # P(k >= 4) = discount ** 3; a running episode at the buffer head is never
  # absorbed.
  buf = TrajectoryBuffer(64, 1, 1, 1, 1, "cpu")
  for t in range(42):
    x = torch.full((1, 1), float(t))
    buf.add(x, x, x, torch.tensor([(t + 1) % 5 == 0]))
  s, _, g = buf.sample(200_000, discount=0.8)
  first = (s % 5 == 0) & (s < 40)
  k = (g - s)[first]
  assert torch.all(k <= 4)
  assert float((k == 4).float().mean()) == pytest.approx(0.8**3, abs=0.01)
  # Episode 40-41 is running: its goals are true later states only.
  assert torch.all(g[s == 40] == 41)
  assert not torch.any(s == 41)


def test_future_offset_is_truncated_geometric():
  # One long episode: the offset follows geometric(1 - discount) on 1..avail.
  buf = TrajectoryBuffer(4000, 1, 1, 1, 1, "cpu")
  for t in range(4000):
    x = torch.full((1, 1), float(t))
    buf.add(x, x, x, torch.tensor([False]))
  s, _, g = buf.sample(200_000, discount=0.95)
  k = (g - s)[s < 3000]  # far from the end: truncation is negligible
  assert float(k.mean()) == pytest.approx(1.0 / (1.0 - 0.95), rel=0.03)


def test_tanh_actor_log_prob_matches_torch():
  torch.manual_seed(0)
  actor = TanhGaussianActor(3, 2, 4, (16,), min_std=1e-6)
  s, g = torch.randn(64, 3), torch.randn(64, 2)
  action, log_prob = actor.sample(s, g)
  mean, std = actor._dist(s, g)
  dist = torch.distributions.TransformedDistribution(
    torch.distributions.Normal(mean, std), [torch.distributions.TanhTransform()]
  )
  expected = dist.log_prob(action.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1)
  torch.testing.assert_close(log_prob, expected, atol=2e-3, rtol=1e-3)
  assert torch.all(action.abs() < 1.0)


@pytest.fixture(scope="module")
def sgcrl_env():
  import wuji_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(_TASK, play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.max_common_steps = 1000
  yield env
  env.close()


def test_sgcrl_groups_are_goal_free_and_aligned(sgcrl_env):
  from wuji_mjlab.tasks.reorient import mdp

  obs, _ = sgcrl_env.reset()
  cmd = sgcrl_env.command_manager.get_term("reorient_command")
  assert obs["crl_goal"].shape[-1] == obs["crl_target"].shape[-1] == 9
  # The achieved goal of a state at the commanded pose equals the target.
  pos = mdp.cube_pos_in_tag(sgcrl_env)
  torch.testing.assert_close(obs["crl_goal"][:, :3], pos)
  target = obs["crl_target"]
  torch.testing.assert_close(target[:, :3], cmd.goal_pos_tag)
  torch.testing.assert_close(target[:, 3:], mdp.rot_6d(cmd.goal_quat_tag))
  # Moving the goal must not change the state the critic and actor see.
  state = sgcrl_env.observation_manager.compute_group("crl_state").clone()
  cmd.goal_pos_tag += 0.05
  cmd.goal_quat_tag[:] = torch.tensor([math.cos(0.5), math.sin(0.5), 0.0, 0.0])
  torch.testing.assert_close(
    sgcrl_env.observation_manager.compute_group("crl_state"), state
  )


def test_trainer_acts_and_updates(sgcrl_env, tmp_path):
  from wuji_mjlab.rl.sgcrl import SgcrlCfg, SgcrlTrainer

  cfg = SgcrlCfg(
    max_iterations=4,
    steps_per_iteration=4,
    updates_per_iteration=2,
    batch_size=8,
    random_action_steps=4,
    buffer_steps=32,
    hidden_dims=(32,),
    log_interval=2,
    save_interval=4,
  )
  trainer = SgcrlTrainer(sgcrl_env, cfg, str(tmp_path), "cpu")
  trainer.learn()
  assert (tmp_path / "model_4.pt").exists()
  action = trainer.act(sgcrl_env.observation_manager.compute())
  assert action.shape == (2, sgcrl_env.action_manager.total_action_dim)
  assert torch.all(action.abs() <= 1.0)


def test_contact_error_is_zero_on_target_and_in_the_object_frame():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient import mdp

  # Object turned 90 deg about z; one target at (0.1, 0, 0) with normal +x
  # (world), so the touching tip reference point sits 5 mm further out.
  quat = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])

  def error(tip):
    command = SimpleNamespace(
      fingertip_target_pos_w=torch.tensor([[[0.1, 0.0, 0.0]]]),
      fingertip_target_dir_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
      fingertip_pos_w=torch.tensor([[tip]]),
      object=SimpleNamespace(data=SimpleNamespace(root_link_quat_w=quat)),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    return mdp.fingertip_contact_error(env, "reorient_command", pad_depth=0.005)

  torch.testing.assert_close(error([0.105, 0.0, 0.0]), torch.zeros(1, 3))
  # 1 cm off along world +y is object +x after the 90 deg turn about z.
  torch.testing.assert_close(
    error([0.105, 0.01, 0.0]), torch.tensor([[0.01, 0.0, 0.0]]), atol=1e-6, rtol=0
  )


def test_sgcrl_contact_groups():
  import wuji_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("WujiHand2_ReorientStaplerFixedGoalWrist_SGCRLContact_50Hz")
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    base = load_env_cfg(_TASK)
    assert obs["crl_goal"].shape[-1] == obs["crl_target"].shape[-1] == 9 + 15
    assert torch.all(obs["crl_target"][:, 9:] == 0.0)
    state_dim = obs["crl_state"].shape[-1]
    assert "fingertip_targets" in cfg.observations["crl_state"].terms
    assert "fingertip_targets" not in base.observations["crl_state"].terms
    assert state_dim > 30
  finally:
    env.close()
