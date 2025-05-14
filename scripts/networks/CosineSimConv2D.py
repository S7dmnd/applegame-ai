import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimConv2D(nn.Module):
    def __init__(self, out_channels, kernel_size, embedding_dim, padding='same', in_channels=1 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.kernel_dim = embedding_dim * kernel_size * kernel_size
        self.padding = padding

        # Learnable kernel weights (O, D)
        self.weight = nn.Parameter(torch.randn(out_channels, self.kernel_dim))
        self.kernel_size_tuple = (kernel_size, kernel_size)

    def forward(self, x):
        """
        x: Tensor of shape (B, E, H, W)
        """
        B, E, H, W = x.shape
        assert E == self.embedding_dim, "Embedding dim mismatch"

        if self.padding == 'same':
            pad = self.kernel_size // 2
            x = F.pad(x, (pad, pad, pad, pad))  # (left, right, top, bottom)
        elif self.padding == 'valid':
            pad = 0
        else:
            raise ValueError("padding must be 'same' or 'valid'")

        # Sliding patches: (B, E*kh*kw, L)
        x_patches = F.unfold(x, kernel_size=self.kernel_size_tuple)
        x_patches = x_patches.permute(0, 2, 1)  # (B, L, D)
        x_norm = F.normalize(x_patches, dim=2)

        w_norm = F.normalize(self.weight, dim=1)  # (O, D)
        sim = torch.einsum('bld,od->blo', x_norm, w_norm)  # (B, L, O)
        sim = sim.permute(0, 2, 1)  # (B, O, L)

        H_out = H if self.padding == 'same' else H - self.kernel_size + 1
        W_out = W if self.padding == 'same' else W - self.kernel_size + 1
        return sim.view(B, self.out_channels, H_out, W_out)

class CosineSimConv2D_v2(nn.Module):
    def __init__(self, out_channels, kernel_size, embedding_dim, padding='same'):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.gamma = nn.Parameter(torch.tensor(10.0))  # Learnable scaling factor

        # Learnable kernel weights: (O, E, k, k)
        self.weight = nn.Parameter(
            torch.randn(out_channels, embedding_dim, kernel_size, kernel_size)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, E, H, W)
        returns: Tensor of shape (B, O, H_out, W_out)
        """
        B, E, H, W = x.shape
        k = self.kernel_size

        # Pad input
        if self.padding == 'same':
            pad = k // 2
            x = F.pad(x, (pad, pad, pad, pad))
        elif self.padding == 'valid':
            pad = 0
        else:
            raise ValueError("padding must be 'same' or 'valid'")

        # Unfold input patches: (B, E*k*k, L)
        x_patches = F.unfold(x, kernel_size=(k, k))  # (B, E*k*k, L)
        L = x_patches.shape[-1]

        # Reshape to (B, L, E, k, k)
        x_patches = x_patches.view(B, E, k, k, L).permute(0, 4, 1, 2, 3)  # (B, L, E, k, k)

        # Normalize input patch vectors (each (E,) vector)
        x_patches = F.normalize(x_patches, dim=2)

        # Normalize kernel weights per (E,) vector
        w = F.normalize(self.weight, dim=1)  # (O, E, k, k)

        # Scaled cosine similarity via batch einsum
        sim = self.gamma * torch.einsum('bleij,oeij->blo', x_patches, w)  # (B, L, O)
        sim = sim.permute(0, 2, 1)  # (B, O, L)

        H_out = H if self.padding == 'same' else H - k + 1
        W_out = W if self.padding == 'same' else W - k + 1
        return sim.view(B, self.out_channels, H_out, W_out)
