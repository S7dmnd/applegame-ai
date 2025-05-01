from typing import Sequence
import numpy as np
import torch
from torch import nn

from scripts.utils import pytorch_utils as ptu
from scripts.networks.ppo_actor import PPOActor
from scripts.networks.ppo_critic import PPOCritic, PPOQCritic


class PPOAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        baseline_learning_rate: float,
        baseline_gradient_steps: int,
        gae_lambda: float,
        normalize_advantages: bool,
        n_ppo_epochs: int = 4,
        n_ppo_minibatches: int = 4,
        ppo_cliprange: float = 0.2,
    ):
        super().__init__()

        self.actor = PPOActor(obs_dim=ob_dim, act_dim=ac_dim, hidden_size=layer_size, n_layers=n_layers, learning_rate=learning_rate)
        self.critic = PPOCritic(ob_dim, hidden_size=layer_size, n_layers=n_layers, learning_rate=baseline_learning_rate)
        self.baseline_gradient_steps = baseline_gradient_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.n_ppo_epochs = n_ppo_epochs
        self.n_ppo_minibatches = n_ppo_minibatches
        self.ppo_cliprange = ppo_cliprange

    def update(
        self,
        obs: Sequence[torch.Tensor],
        actions: Sequence[np.ndarray],
        rewards_per_traj: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
        next_obs: Sequence[torch.Tensor] = None
    ) -> dict:
        if isinstance(obs, (list, tuple)):
            obs = torch.cat(obs, dim=0)  # shape: (batch_size, 1, H, W)
        actions = np.array(actions)
        rewards = np.concatenate(rewards_per_traj)
        terminals = np.array(terminals).reshape(-1, 1)

        q_values = np.concatenate([self._discounted_reward_to_go(r) for r in rewards_per_traj])
        if isinstance(self.critic, PPOCritic):
            advantages = self._estimate_advantage(obs, rewards, q_values, terminals)
        elif isinstance(self.critic, PPOQCritic):
            advantages, q_targets = self._estimate_advantage(obs, rewards, q_values, terminals, actions=actions, next_obs=next_obs)

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # print(actions.shape)
        logp_old = self._calculate_log_probs(obs, actions)

        n_batch = len(obs)
        inds = np.arange(n_batch)
        for _ in range(self.n_ppo_epochs):
            np.random.shuffle(inds)
            minibatch_size = (n_batch + self.n_ppo_minibatches - 1) // self.n_ppo_minibatches
            for start in range(0, n_batch, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]
                info = self.actor.ppo_update(
                    obs[mb_inds],
                    actions[mb_inds],
                    advantages[mb_inds],
                    logp_old[mb_inds],
                    self.ppo_cliprange,
                )

        for _ in range(self.baseline_gradient_steps):
            if isinstance(self.critic, PPOCritic):
                critic_info = self.critic.update(obs, q_values)
            elif isinstance(self.critic, PPOQCritic):
                critic_info = self.critic.update(obs, actions, q_targets)
        info.update(critic_info)
        return info

    def _estimate_advantage(
        self,
        obs: torch.Tensor,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
        actions: np.ndarray = None,
        next_obs: torch.Tensor = None,
    ) -> np.ndarray:
        if isinstance(self.critic, PPOCritic):
            # obs: (batch, 1, 17, 10)
            obs = obs.to(ptu.device)
            values = self.critic(obs).detach().cpu().numpy()
            batch_size = obs.shape[0]

            values = np.append(values, [0])
            advantages = np.zeros(batch_size + 1)

            for i in reversed(range(batch_size)):
                mask = 1.0 - terminals[i]
                delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
                advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1] * mask

            return advantages[:-1]
        elif isinstance(self.critic, PPOQCritic):
            obs = obs.to(ptu.device)
            actions = ptu.from_numpy(actions)
            next_obs = next_obs.to(ptu.device)

            with torch.no_grad():
                next_actions = torch.cat([
                    ptu.from_numpy(self.actor.get_action(next_obs[i:i+1])[None])  # (1, 4)
                    for i in range(len(next_obs))
                ], dim=0)  # shape: (B, 4)
                q_next = self.critic(next_obs, next_actions)
                q_next = q_next * (1 - ptu.from_numpy(terminals))

            q_target = ptu.from_numpy(rewards) + self.gamma * q_next
            q_pred = self.critic(obs, actions)
            advantages = q_target - q_pred
            return advantages.detach().cpu().numpy(), q_target.detach().cpu().numpy()


    def _discounted_reward_to_go(self, rewards: np.ndarray) -> np.ndarray:
        ret = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            discounted_sum = 0
            discount = 1.0
            for k in range(t, len(rewards)):
                discounted_sum += discount * rewards[k]
                discount *= self.gamma
            ret[t] = discounted_sum
        return ret

    def _calculate_log_probs(self, obs: torch.Tensor, actions: np.ndarray) -> np.ndarray:
        actions = ptu.from_numpy(actions)
        dist = self.actor(obs)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        return ptu.to_numpy(log_probs)
