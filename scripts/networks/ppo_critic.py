import torch
from torch import nn, optim
from torch.nn import functional as F
from scripts.utils import pytorch_utils as ptu
import numpy as np

class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, n_layers=2, learning_rate=3e-4):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # (B, 16, 17, 10)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 32, 17, 10)
            nn.ReLU(),
            nn.Flatten(),                                # (B, 32*17*10)
            nn.Linear(32 * 17 * 10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(ptu.device)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        values = self.value_net(obs).squeeze(-1)  # (batch_size,)
        assert values.ndim == 1
        return values

    def update(self, obs, targets):
        obs = obs.to(ptu.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=ptu.device)

        predictions = self(obs)
        loss = nn.MSELoss()(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"CriticLoss": loss.item()}
