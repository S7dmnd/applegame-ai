import torch
from torch import nn

import scripts.utils.pytorch_utils as ptu

class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)
    
    def forward(self, obs, acs):
        obs = obs.view(obs.size(0), -1)  # flatten CNN obs (B, C, H, W) -> (B, D) 지금은 CNN Critic이 아니라서...
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)
