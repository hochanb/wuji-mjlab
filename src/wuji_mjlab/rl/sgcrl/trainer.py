# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""SGCRL trainer for a ManagerBasedRlEnv (rewards are ignored).

Every rollout commands the single target goal from the ``target_group``
observation. The critic ``phi(s, a)^T psi(g)`` is trained with InfoNCE on
(state, action, discounted future achieved goal) triples with the other
goals in the batch as negatives, plus a LogSumExp regulariser; the actor
maximises the critic for its own future goals and for goals shuffled across
the batch, with an adaptive entropy bonus.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude
from torch.utils.tensorboard import SummaryWriter

from .buffer import TrajectoryBuffer
from .config import SgcrlCfg
from .networks import ContrastiveCritic, RunningNorm, TanhGaussianActor


class SgcrlTrainer:
  def __init__(self, env, cfg: SgcrlCfg, log_dir: str | None, device: str) -> None:
    self.env, self.cfg, self.device = env, cfg, device
    self.log_dir = Path(log_dir) if log_dir else None
    obs, _ = env.reset()
    # Spread the episodes over their length so resets do not come in waves.
    env.episode_length_buf = torch.randint_like(
      env.episode_length_buf, high=int(env.max_episode_length)
    )
    s, g, target = self._split(obs)
    self.state_dim, self.goal_dim = s.shape[-1], g.shape[-1]
    if target.shape[-1] != self.goal_dim:
      raise ValueError(
        f"SGCRL: target goal has {target.shape[-1]} dims, achieved goal {self.goal_dim}."
      )
    self.action_dim = env.action_manager.total_action_dim
    self.num_envs = env.num_envs
    self._obs = obs

    hidden = tuple(cfg.hidden_dims)
    self.actor = TanhGaussianActor(
      self.state_dim, self.goal_dim, self.action_dim, hidden, cfg.actor_min_std
    ).to(device)
    self.critic = ContrastiveCritic(
      self.state_dim, self.action_dim, self.goal_dim, hidden, cfg.repr_dim
    ).to(device)
    self.state_norm = RunningNorm(self.state_dim).to(device)
    self.goal_norm = RunningNorm(self.goal_dim).to(device)
    self.log_alpha = torch.zeros((), device=device, requires_grad=True)
    self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
    self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
    self.buffer = TrajectoryBuffer(
      cfg.buffer_steps,
      self.num_envs,
      self.state_dim,
      self.action_dim,
      self.goal_dim,
      device,
    )
    self.iteration = 0
    self.writer = SummaryWriter(str(self.log_dir)) if self.log_dir else None
    self._ep_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
    self._ep_min_ori = torch.full((self.num_envs,), torch.inf, device=device)
    self._ep_min_pos = torch.full((self.num_envs,), torch.inf, device=device)

  # -- observations -----------------------------------------------------------
  def _split(self, obs):
    c = self.cfg
    return obs[c.state_group], obs[c.goal_group], obs[c.target_group]

  def _normed(self, state, goal):
    if not self.cfg.normalize_obs:
      return state, goal
    return self.state_norm(state), self.goal_norm(goal)

  @torch.no_grad()
  def act(self, obs, deterministic: bool = False) -> torch.Tensor:
    """Action toward the commanded (single) goal."""
    s, _, target = self._split(obs)
    s, target = self._normed(s, target)
    if deterministic:
      return self.actor.mode(s, target)
    return self.actor.sample(s, target)[0]

  # -- losses -----------------------------------------------------------------
  def _update(self) -> dict[str, float]:
    c = self.cfg
    s, a, g = self.buffer.sample(c.batch_size, c.discount, c.absorb_episode_end)
    s, g = self._normed(s, g)
    b = s.shape[0]
    labels = torch.arange(b, device=self.device)

    # Critic: InfoNCE over the batch's future goals + LogSumExp regulariser.
    logits = self.critic(s, a, g)
    lse = torch.logsumexp(logits, dim=1)
    critic_loss = (
      torch.nn.functional.cross_entropy(logits, labels)
      + c.logsumexp_coef * (lse**2).mean()
    )
    self.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    self.critic_opt.step()

    # Actor: own future goal for a fraction of the batch, a shuffled one else.
    n_own = int(round(c.actor_own_goal_frac * b))
    goals = torch.cat((g[:n_own], torch.roll(g, 1, dims=0)[n_own:]))
    action, log_prob = self.actor.sample(s, goals)
    alpha = self.log_alpha.exp().detach()
    q = self.critic.paired(s, action, goals)
    actor_loss = (alpha * log_prob - q).mean()
    self.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    self.actor_opt.step()

    alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + c.target_entropy)).mean()
    self.alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    self.alpha_opt.step()

    with torch.no_grad():
      eye = torch.eye(b, device=self.device, dtype=torch.bool)
      return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "categorical_accuracy": (logits.argmax(1) == labels).float().mean().item(),
        "logits_pos": logits[eye].mean().item(),
        "logits_neg": logits[~eye].mean().item(),
        "logsumexp": (lse**2).mean().item(),
        "alpha": alpha.item(),
        "entropy": -log_prob.mean().item(),
      }

  # -- rollout ----------------------------------------------------------------
  def _episode_stats(self, done: torch.Tensor, stats) -> None:
    cmd = self.env.command_manager.get_term("reorient_command")
    if done.any():
      stats["success"].append(self._ep_reached[done].float())
      stats["min_ori_error"].append(self._ep_min_ori[done])
      stats["min_pos_error"].append(self._ep_min_pos[done])
      self._ep_reached[done] = False
      self._ep_min_ori[done] = torch.inf
      self._ep_min_pos[done] = torch.inf
    live = ~done
    ori = quat_error_magnitude(cmd.object.data.root_link_quat_w, cmd.goal_quat_w)
    self._ep_reached |= cmd.success_achieved & live
    self._ep_min_ori = torch.where(
      live, torch.minimum(self._ep_min_ori, ori), self._ep_min_ori
    )
    self._ep_min_pos = torch.where(
      live, torch.minimum(self._ep_min_pos, cmd.pos_error()), self._ep_min_pos
    )

  @torch.no_grad()
  def _collect(self, random: bool, stats) -> None:
    for _ in range(self.cfg.steps_per_iteration):
      s, g, _ = self._split(self._obs)
      if random:
        action = (
          2.0 * torch.rand(self.num_envs, self.action_dim, device=self.device) - 1.0
        )
      else:
        action = self.act(self._obs)
      if self.cfg.normalize_obs:
        self.state_norm.update(s)
        self.goal_norm.update(g)
      self._obs, _, terminated, truncated, extras = self.env.step(action)
      done = (terminated | truncated).bool()
      self.buffer.add(s, action, g, done)
      self._episode_stats(done, stats)
      for k, v in extras.get("log", {}).items():
        stats["env/" + k].append(torch.as_tensor(v, dtype=torch.float32).reshape(-1))

  def learn(self) -> None:
    c = self.cfg
    random_iters = -(-c.random_action_steps // c.steps_per_iteration)
    t0 = time.time()
    stats = defaultdict(list)
    while self.iteration < c.max_iterations:
      random = self.iteration < random_iters
      self._collect(random, stats)
      # A resumed run starts with an empty buffer: refill it before updating.
      if not random and self.buffer.size >= c.random_action_steps:
        for _ in range(c.updates_per_iteration):
          for k, v in self._update().items():
            stats["train/" + k].append(torch.tensor([v]))
      self.iteration += 1
      if self.iteration % c.log_interval == 0:
        self._log(stats, time.time() - t0)
        stats, t0 = defaultdict(list), time.time()
      if self.log_dir and self.iteration % c.save_interval == 0:
        self.save(self.log_dir / f"model_{self.iteration}.pt")
    if self.log_dir:
      self.save(self.log_dir / f"model_{self.iteration}.pt")

  def _log(self, stats, elapsed: float) -> None:
    c = self.cfg
    env_steps = self.buffer.step * self.num_envs
    fps = c.log_interval * c.steps_per_iteration * self.num_envs / max(elapsed, 1e-6)
    line = [f"it {self.iteration:6d}  env steps {env_steps:.3e}  fps {fps:8.0f}"]
    for k in sorted(stats):
      vals = torch.cat([v.float().cpu() for v in stats[k]])
      if vals.numel() == 0:
        continue
      mean = float(vals.mean())
      if self.writer:
        self.writer.add_scalar(k, mean, self.iteration)
      if k.startswith(("train/", "success", "min_")) or k.endswith(
        ("ori_error", "pos_error", "cage_drop", "time_out")
      ):
        line.append(f"{k} {mean:.4f}")
    if self.writer:
      self.writer.add_scalar("perf/fps", fps, self.iteration)
    print("  ".join(line), flush=True)

  # -- checkpoints ------------------------------------------------------------
  def save(self, path: Path) -> None:
    torch.save(
      {
        "iteration": self.iteration,
        "cfg": dataclasses.asdict(self.cfg),
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict(),
        "state_norm": self.state_norm.state_dict(),
        "goal_norm": self.goal_norm.state_dict(),
        "log_alpha": self.log_alpha.detach(),
        "actor_opt": self.actor_opt.state_dict(),
        "critic_opt": self.critic_opt.state_dict(),
        "alpha_opt": self.alpha_opt.state_dict(),
      },
      path,
    )

  def load(self, path: str, actor_only: bool = False) -> None:
    ckpt = torch.load(path, map_location=self.device)
    self.actor.load_state_dict(ckpt["actor"])
    self.state_norm.load_state_dict(ckpt["state_norm"])
    self.goal_norm.load_state_dict(ckpt["goal_norm"])
    if actor_only:
      return
    self.critic.load_state_dict(ckpt["critic"])
    with torch.no_grad():
      self.log_alpha.copy_(ckpt["log_alpha"])
    self.actor_opt.load_state_dict(ckpt["actor_opt"])
    self.critic_opt.load_state_dict(ckpt["critic_opt"])
    self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
    self.iteration = int(ckpt["iteration"])
