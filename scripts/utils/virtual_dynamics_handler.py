import numpy as np
import pandas as pd
import torch
from scripts.utils import pytorch_utils as ptu
from scripts.utils import env_utils
import math

class VirtualDynamicsHandler:
    def __init__(self, grid_shape=(17, 10), max_episode_size=50, seed=None):
        self.grid_shape = grid_shape  # (width, height)
        self.max_episode_size = max_episode_size
        self.current_grid = None
        self.previous_score = 0
        self.current_score = 0
        self.done = False
        self.step_count = 0
        self.seed = seed

        self.all_actions =[]

        for col1 in range(self.grid_shape[0]):
            for row1 in range(self.grid_shape[1]):
                for col2 in range(col1, self.grid_shape[0]):
                    for row2 in range(row1, self.grid_shape[1]):
                        self.all_actions.append(((col1, row1), (col2, row2)))

        ptu.init_gpu()

    def reset(self):
        """환경 초기화: 1~9 사이 숫자로 가득 찬 Grid 생성"""
        if self.seed:
            np.random.seed(self.seed)
        grid_array = np.random.randint(1, 10, size=(self.grid_shape[1], self.grid_shape[0]))
        self.current_grid = pd.DataFrame(grid_array)
        self.previous_score = 0
        self.current_score = 0
        self.done = False
        self.step_count = 0
        return torch.tensor(self.current_grid.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(ptu.device)

    def perform_action(self, action):
        """
        action: ((col1, row1), (col2, row2)) 좌표를 받아서 박스를 형성한 뒤,
        박스 내 값의 합이 10이면
        해당 박스 내부 값을 모두 0으로 만듦 (사과 없앰)
        """
        col1, row1, col2, row2 = np.floor(action).astype(int)
        if col1 <= col2:
            small_col = col1
            large_col = col2 + 1
            # (col1 = 0, col2 = 2) 이라면 [0, 1, 2] = iloc[1:3] 선택
        else:
            small_col = col2 + 1
            large_col = col1
            # (col1 = 2, col2 = 0) 이라면 [1] = iloc[1:2] 선택
        
        if row1 <= row2:
            small_row = row1
            large_row = row2 + 1
        else:
            small_row = row2 + 1
            large_row = row1

        affected_area = self.current_grid.iloc[small_row:large_row, small_col:large_col]
        self.previous_score = self.current_score
        if affected_area.values.sum() == 10:
            self.current_grid.iloc[small_row:large_row, small_col:large_col] = 0
        self.current_score = (self.current_grid == 0).sum().sum()

        reward = self.compute_reward(affected_area.values.sum())

        return reward
    
    def compute_reward(self, box_sum):
        diff = abs(box_sum - 10)
        reward = 20 * math.exp(-diff**2 / 2.0) - 10  # max: +10, min: -10
        return reward

    def step(self, action):
        # Action index -> Action으로 변환
        # action_vec: np.ndarray of shape (4,)
        reward = self.perform_action(action)
        self.step_count += 1
        self.done = self.step_count >= self.max_episode_size or self._check_done()
        # flatten_grid = self.current_grid.to_numpy().flatten() # flatten한 것 (170, )
        # print(flatten_grid)
        torch_grid = torch.tensor(self.current_grid.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(ptu.device)
        return torch_grid, reward, self.done

    def _check_done(self):
        return (self.current_grid == 0).all().all()
    
    def _index_to_action(self, action_index):
        return self.all_actions[action_index]

    def render(self):
        print(self.current_grid.to_string(index=False, header=False))
