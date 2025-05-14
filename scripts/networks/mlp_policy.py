from typing import Optional

from torch import nn
import torch.nn.functional as F
import torch
from torch import distributions

from scripts.utils import pytorch_utils as ptu
from scripts.utils.distributions import make_tanh_transformed, make_multi_normal
from scripts.networks.CosineSimConv2D import CosineSimConv2D, CosineSimConv2D_v2

class MLPPolicy(nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )


    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # obs.shape = (B, 1, 17, 10) --> Flatten
        obs = obs.view(obs.size(0), -1)  # (B, 170)
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution

class CNNPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_shape: tuple[int, int, int],  # (C, H, W)
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = True,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        C, H, W = ob_shape
        print(f"Initializing CNN Actor... Ob Shape = {ob_shape}")

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(ptu.device)

        conv_out_dim = 32 * H * W

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=conv_out_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=2 * ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        obs: (B, C, H, W) Tensor
        """
        h = self.cnn(obs)  # shape: (B, conv_out_dim)

        if self.discrete:
            logits = self.logits_net(h)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(h), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(h)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                action_distribution = make_multi_normal(mean, std)

        return action_distribution

class EnhancedCNNPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_shape: tuple[int, int, int],  # (C, H, W)
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = True,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        C, H, W = ob_shape
        print(f"Initializing Enhanced CNN Actor... Ob Shape = {ob_shape}")

        # Multi-Kernel block
        self.conv3 = nn.Conv2d(C, 32, kernel_size=3, padding=1).to(ptu.device)
        self.conv5 = nn.Conv2d(C, 32, kernel_size=5, padding=2).to(ptu.device)
        self.conv7 = nn.Conv2d(C, 32, kernel_size=7, padding=3).to(ptu.device)

        self.relu = nn.ReLU()

        # Mixing conv layers
        self.conv_mix = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        ).to(ptu.device)

        self.flatten = nn.Flatten()
        conv_out_dim = 64 * H * W  # after conv_mix: 64 channels

        # Fully connected head
        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=conv_out_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=2 * ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        obs: (B, C, H, W)
        """
        x3 = self.relu(self.conv3(obs))
        x5 = self.relu(self.conv5(obs))
        x7 = self.relu(self.conv7(obs))
        x = torch.cat([x3, x5, x7], dim=1)  # (B, 96, H, W)

        h = self.conv_mix(x)                # (B, 64, H, W)
        h = self.flatten(h)                 # (B, 64 * H * W)

        if self.discrete:
            logits = self.logits_net(h)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(h), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(h)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                action_distribution = make_multi_normal(mean, std)

        return action_distribution

class EmbeddingCNNPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_shape: tuple[int, int, int],  # (C, H, W)
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = True,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
        emb_dim: int = 16,
        num_categories: int = 10,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        _, H, W = ob_shape
        self.embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=emb_dim).to(ptu.device)

        print(f"Initializing Embedding CNN Actor... Observation Shape = {ob_shape}, Embedding dim = {emb_dim}")

        # CNN Block
        self.cnn = nn.Sequential(
            nn.Conv2d(emb_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        ).to(ptu.device)

        self.flatten = nn.Flatten()
        conv_out_dim = 64 * H * W

        # Fully connected head
        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=conv_out_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=2 * ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.Tensor) -> distributions.Distribution:
        """
        obs: (B, H, W) — categorical integer grid
        """
        obs = obs.to(ptu.device)
        obs = obs.squeeze(1)  # (B, H, W), 채널 차원 삭제
        x = self.embedding(obs.long())  # (B, H, W, E)
        x = x.permute(0, 3, 1, 2)       # (B, E, H, W)
        x = self.cnn(x)                 # (B, 64, H, W)
        h = self.flatten(x)             # (B, 64 * H * W)

        if self.discrete:
            logits = self.logits_net(h)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(h), 2, dim=-1)
                std = F.softplus(std) + 1e-2
            else:
                mean = self.net(h)
                if self.fixed_std:
                    std = self.std
                else:
                    std = F.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                action_distribution = make_multi_normal(mean, std)

        return action_distribution

class CosSimEmbeddingPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_shape: tuple[int, int, int],  # (C, H, W)
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = True,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
        emb_dim: int = 16,
        num_categories: int = 10,
        cos_out_channels: int = 64,
        kernel_size: int = 3,
        use_embedding_mapping: bool = False,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        print(f"Initializing Custom CNN Actor... Observation Shape = {ob_shape}, Embedding dim = {emb_dim}")
        _, H, W = ob_shape

        # embedding 매핑 분기
        emb_dim = emb_dim if use_embedding_mapping else num_categories
        self.num_categories = num_categories
        self.emb_dim = emb_dim
        self.use_embedding_mapping = use_embedding_mapping
        if use_embedding_mapping:
            self.embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=emb_dim).to(ptu.device)

        self.cosine_conv = CosineSimConv2D_v2(
            out_channels=cos_out_channels,
            kernel_size=kernel_size,
            embedding_dim=emb_dim,
            padding='same'
        ).to(ptu.device)

        self.flatten = nn.Flatten()
        conv_out_dim = cos_out_channels * H * W

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=conv_out_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=2 * ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=conv_out_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.Tensor) -> distributions.Distribution:
        obs = obs.to(ptu.device).squeeze(1)  # (B, H, W)
        x = self.embedding(obs.long()) if self.use_embedding_mapping else F.one_hot(obs.long(), num_classes=self.num_categories).float() # (B, H, W, E)
        x = x.permute(0, 3, 1, 2)            # (B, E, H, W)
        x = self.cosine_conv(x)              # (B, O, H, W)
        h = self.flatten(x)

        if self.discrete:
            logits = self.logits_net(h)
            return distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(h), 2, dim=-1)
                std = F.softplus(std) + 1e-2
            else:
                mean = self.net(h)
                std = self.std if self.fixed_std else F.softplus(self.std) + 1e-2

            if self.use_tanh:
                return make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)
            
            