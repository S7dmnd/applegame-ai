from typing import Sequence
import numpy as np
import torch
from torch import nn

from scripts.utils import pytorch_utils as ptu
from scripts.networks.ppo_actor import PPOActor
from scripts.networks.ppo_critic import PPOCritic


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

        self.actor = PPOActor(ob_dim, ac_dim, layer_size, n_layers, learning_rate)
        self.critic = PPOCritic(ob_dim, n_layers, layer_size, baseline_learning_rate)
        self.baseline_gradient_steps = baseline_gradient_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.n_ppo_epochs = n_ppo_epochs
        self.n_ppo_minibatches = n_ppo_minibatches
        self.ppo_cliprange = ppo_cliprange

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards_per_traj: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        obs = np.vstack(obs)
        actions = np.array(actions).reshape(-1, 1) # 각 action이 그냥 int라서 그런듯?
        rewards = np.concatenate(rewards_per_traj)
        terminals = np.array(terminals).reshape(-1, 1)

        q_values = np.concatenate([self._discounted_reward_to_go(r) for r in rewards_per_traj])
        advantages = self._estimate_advantage(obs, rewards, q_values, terminals)

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logp_old = self._calculate_log_probs(obs, actions).reshape(-1)
        
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
            critic_info = self.critic.update(obs, q_values)
        info.update(critic_info)
        return info

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        obs_torch = ptu.from_numpy(obs)
        values = self.critic(obs_torch).detach().cpu().numpy()
        batch_size = obs.shape[0]

        values = np.append(values, [0])
        advantages = np.zeros(batch_size + 1)

        for i in reversed(range(batch_size)):
            mask = 1.0 - terminals[i]
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1] * mask

        return advantages[:-1]

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

    def _calculate_log_probs(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        dist = self.actor(obs)
        log_probs = dist.log_prob(actions)
        return ptu.to_numpy(log_probs)
