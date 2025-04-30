from torch import nn
from torch import optim
from torch import distributions
from scripts.utils import pytorch_utils as ptu
import torch

import numpy as np

class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128, n_layers=2, learning_rate=3e-4):
        super().__init__()
        self.policy_net = ptu.build_mlp(obs_dim, act_dim, n_layers, hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def forward(self, obs):
        logits = self.policy_net(obs)
        return distributions.Categorical(logits=logits)  # R^4 Discrete Action

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = ptu.from_numpy(obs[None])  # (1, obs_dim)
        dist = self.forward(obs_tensor)
        action = dist.sample()
        return ptu.to_numpy(action)[0]

    def update(self, obs, actions, advantages):
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        dist = self.forward(obs)
        logp = dist.log_prob(actions)
        loss = -(logp * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'ActorLoss': ptu.to_numpy(loss)}
    
    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions).squeeze(-1)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp).squeeze(-1)

        dist = self.forward(obs)
        logp = dist.log_prob(actions)

        #print(f"[ppo_update] obs: {obs.shape}, actions: {actions.shape}, advantages: {advantages.shape}")
        #print(f"[ppo_update] logp: {logp.shape}, old_logp: {old_logp.shape}")

        ratio = torch.exp(logp - old_logp)

        clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)
        unclipped = ratio * advantages
        clipped = clipped_ratio * advantages
        loss = -torch.mean(torch.min(unclipped, clipped))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"PPO Loss": ptu.to_numpy(loss)}

