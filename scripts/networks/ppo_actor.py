from torch import nn
from torch import optim
from torch import distributions
from scripts.utils import pytorch_utils as ptu
import torch

import numpy as np

class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim=4, hidden_size=128, n_layers=2, learning_rate=3e-4):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 17 * 10, act_dim),  # 32 channels * H * W
            nn.Sigmoid()  # → [0, 1] 출력
        ).to(ptu.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.action_scale = torch.tensor([17.0, 10.0, 17.0, 10.0], device=ptu.device)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=ptu.device))

    def forward(self, obs):
        raw = self.policy_net(obs)  # shape: (B, 4), unbounded
        mean = torch.sigmoid(raw) * self.action_scale  # scale to [0, max]
        std = torch.exp(self.log_std).expand_as(mean)
        return distributions.Normal(mean, std)  # or MultivariateNormal if you prefer

    @torch.no_grad()
    def get_action(self, obs:torch.Tensor) -> np.ndarray:
        #obs = torch.Tensor (1, 1, h, w)
        dist = self.forward(obs)
        action = dist.sample()
        return ptu.to_numpy(action)[0]

    def update(self, obs, actions, advantages):
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        dist = self.forward(obs)
        logp = dist.log_prob(actions).sum(axis=-1)  # shape: (B,)
        loss = -(logp * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'ActorLoss': ptu.to_numpy(loss)}
    
    def ppo_update(
        self,
        obs: torch.Tensor,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        actions = ptu.from_numpy(actions)
        assert actions.ndim == 2 and actions.shape[1] == 4, f"Got action shape: {actions.shape}"

        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        dist = self.forward(obs)
        logp = dist.log_prob(actions).sum(axis=-1)  # shape: (B,)

        #print(f"[ppo_update] obs: {obs.shape}, actions: {actions.shape}, advantages: {advantages.shape}")
        #print(f"[ppo_update] logp: {logp.shape}, old_logp: {old_logp.shape}")

        ratio = torch.exp(logp - old_logp)
        #print(advantages.size())

        clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)
        unclipped = ratio * advantages
        clipped = clipped_ratio * advantages
        loss = -torch.mean(torch.min(unclipped, clipped))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"PPO Loss": ptu.to_numpy(loss)}

