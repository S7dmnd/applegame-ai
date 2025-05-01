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

class PPOQCritic(nn.Module):
    def __init__(self, ob_dim, act_dim, n_layers=2, hidden_size=128, learning_rate=1e-3):
        super().__init__()
        self.act_dim = act_dim
        self.ob_dim = ob_dim

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ob_dim + act_dim, hidden_size),
            nn.ReLU(),
            *[
                layer for _ in range(n_layers - 1)
                for layer in (nn.Linear(hidden_size, hidden_size), nn.ReLU())
            ],
            nn.Linear(hidden_size, 1),
        ).to(ptu.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, obs, actions):
        # obs: (B, 1, 17, 10), actions: (B, act_dim)
        obs_flat = obs.view(obs.size(0), -1)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # (B,) → (B, 1)
        x = torch.cat([obs_flat, actions], dim=-1)
        return self.model(x).squeeze(-1)  # (B,)

    def update(self, obs, actions, q_targets):
        pred_q = self.forward(obs, actions)
        loss = nn.functional.mse_loss(pred_q, ptu.from_numpy(q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'QCriticLoss': ptu.to_numpy(loss)}
    
