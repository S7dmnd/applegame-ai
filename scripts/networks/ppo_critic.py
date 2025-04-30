import torch
from torch import nn, optim
from torch.nn import functional as F
from scripts.utils import pytorch_utils as ptu
import numpy as np

class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, n_layers=2, learning_rate=3e-4):
        super().__init__()
        self.value_net = ptu.build_mlp(obs_dim, 1, n_layers, hidden_size)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        values = self.value_net(obs).squeeze(-1)  # (batch_size,)
        assert values.ndim == 1
        return values

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)        # (batch_size, obs_dim)
        q_values = ptu.from_numpy(q_values)  # (batch_size,)
        values = self.forward(obs)

        loss = F.mse_loss(values, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"CriticLoss": ptu.to_numpy(loss)}
