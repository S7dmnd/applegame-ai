import numpy as np
import pandas as pd
import torch
from scripts.utils import pytorch_utils as ptu
from scripts.utils import env_utils
from gymnasium.spaces import Box
import math

class VirtualDynamicsHandler:
    def __init__(self, allow_zeros_in_grid=False, discrete_action=False, grid_shape=(17, 10), max_episode_size=200, seed=None, use_tanh=True, use_one_hot_encoding=False):
        self.grid_shape = grid_shape  # (width, height)
        self.max_episode_size = max_episode_size
        self.current_grid = None
        self.previous_score = 0
        self.current_score = 0
        self.done = False
        self.step_count = 0
        self.seed = seed
        self.discrete_action = discrete_action
        self.allow_zeros_in_grid = allow_zeros_in_grid
        self.use_tanh = use_tanh
        self.use_one_hot_encoding = use_one_hot_encoding
        self.channel = 10 if self.use_one_hot_encoding else 1

        self.all_actions =[]

        for col1 in range(self.grid_shape[0]):
            for row1 in range(self.grid_shape[1]):
                for col2 in range(col1, self.grid_shape[0]):
                    for row2 in range(row1, self.grid_shape[1]):
                        self.all_actions.append(((col1, row1), (col2, row2)))

        ptu.init_gpu()

        self.spec = type("Spec", (), {"max_episode_steps": self.max_episode_size})()
        self.observation_space = Box(
            low=0, high=9, shape=(self.channel, self.grid_shape[1], self.grid_shape[0]), dtype=np.float32
        )
        self.action_space = Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.grid_shape[0], self.grid_shape[1], self.grid_shape[0], self.grid_shape[1]], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        """환경 초기화 및 observation 반환 (np.ndarray)"""
        if self.seed:
            np.random.seed(self.seed)
        if self.allow_zeros_in_grid:
            grid_array = np.random.randint(0, 10, size=(self.grid_shape[1], self.grid_shape[0]))
        else:
            grid_array = np.random.randint(1, 10, size=(self.grid_shape[1], self.grid_shape[0]))
        self.current_grid = pd.DataFrame(grid_array)
        self.previous_score = 0
        self.current_score = 0
        self.done = False
        self.step_count = 0
        return self.get_observation(), {}
    
    def perform_action(self, action):
        """
        action: ((col1, row1), (col2, row2)) 좌표를 받아서 박스를 형성한 뒤,
        박스 내 값의 합이 10이면
        해당 박스 내부 값을 모두 0으로 만듦 (사과 없앰)
        """

        if self.discrete_action:
            assert isinstance(action, (int, np.int))
            col1, row1, col2, row2 = self.all_actions[action] #discrete action perform
        else:
            col1, row1, col2, row2 = np.floor(action).astype(int) # continuous action perform
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

        if isinstance(action, np.ndarray):
            max_abs = np.abs(action).max()
            if max_abs > 17.0 + 1e-3:  # 약간의 수치 오차 허용
                # print("⚠️ [ENV DEBUG] Action outside of [-1, 1]:", action)
                print("⚠️ [ENV DEBUG] Max abs value:", max_abs)
                # raise ValueError(f"[ENV ERROR] Action out of [-1, 1]: {action} (max abs: {max_abs})")
        else:
            print("⚠️ [ENV DEBUG] Action is not np.ndarray:", type(action))
        # print(f"Debugging - Action: {action} / BoxSum: {affected_area.values.sum()} / Reward: {reward}")

        return reward
    
    def get_observation(self) -> np.ndarray:
        """현재 그리드 상태를 (1, H, W) 형태의 np.ndarray로 반환"""
        ob = self.current_grid.values.astype(np.float32)[np.newaxis, ...]
        if self.use_one_hot_encoding:
            ob = self.encode_grid(ob)
        return ob
    
    def compute_reward(self, box_sum):
        diff = abs(box_sum - 10)
        if diff == 0:
            reward = 100 # 정답이면 100
        else: 
            # reward = 2.0 * math.exp((-diff**2) / (2.0 * 10.0 ** 2)) - 1.0 # [-1, 1]
            reward = 1.0 * math.exp((-diff**2) / (2.0 * 10.0 ** 2)) - 1.0  # [-1, 0]
        return reward

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """action: np.ndarray of shape (4,), returns: (obs, reward, done)"""
        reward = self.perform_action(action)
        self.step_count += 1
        self.done = self.step_count >= self.max_episode_size or self._check_done()
        
        info = {}
        if self.done:
            info["episode"] = {
                "r": self.current_score,     # 누적 보상 또는 점수 등
                "l": self.step_count         # 에피소드 길이
            }

        return self.get_observation(), reward, self.done, self.done, info
    
    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        ########## DEBUG ############
        if isinstance(action, np.ndarray):
            max_abs = np.abs(action).max()
            if max_abs > 1.0 + 1e-3:  # 약간의 수치 오차 허용
                # print("⚠️ [ENV DEBUG] Action outside of [-1, 1]:", action)
                print("⚠️ [ENV DEBUG] Max abs value:", max_abs)
                raise ValueError(f"[ENV ERROR] Action out of [-1, 1]: {action} (max abs: {max_abs})")
        else:
            print("⚠️ [ENV DEBUG] Action is not np.ndarray:", type(action))

        low = np.array([0, 0, 0, 0])
        high = np.array([
            self.grid_shape[0], self.grid_shape[1],
            self.grid_shape[0], self.grid_shape[1]
        ])
        return low + 0.5 * (action + 1.0) * (high - low)
    
    def encode_grid(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (1, H, W) 형태의 np.ndarray. 값은 0~9 (0은 빈칸).
        return: (10, H, W) 형태의 one-hot+mask encoding 결과
            - channel 0~8: 숫자 1~9의 one-hot
            - channel 9: 빈 칸 (0) 마스크
        """
        assert obs.ndim == 3 and obs.shape[0] == 1, f"Expected shape (1, H, W), got {obs.shape}"
        _, H, W = obs.shape
        encoded = np.zeros((10, H, W), dtype=np.float32)

        for v in range(1, 10):
            encoded[v-1] = (obs[0] == v).astype(np.float32)
        encoded[9] = (obs[0] == 0).astype(np.float32)  # 0 마스크

        return encoded

    def _check_done(self):
        return (self.current_grid == 0).all().all()
    
    def _index_to_action(self, action_index):
        return self.all_actions[action_index]

    def render(self):
        print(self.current_grid.to_string(index=False, header=False))

    def close(self):
        pass  # 아무것도 안 해도 괜찮음
