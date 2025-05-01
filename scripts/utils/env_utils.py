import math
import torch
import numpy as np
import pandas as pd

def compute_reward(box_sum):
        diff = abs(box_sum - 10)
        reward = 20 * math.exp(-diff**2 / 2.0) - 10  # max: +10, min: -10
        return reward

def to_tensor(grid, device):
    arr = grid.to_numpy(dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)