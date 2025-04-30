import random
from scripts.utils.dynamics_handler import DynamicsHandler
from scripts.agents.simple_digit_classifier import SimpleDigitClassifier
import time
import pyautogui
import torch

def generate_random_action(grid_shape):
    rows, cols = grid_shape
    r1 = random.randint(0, rows - 1)
    c1 = random.randint(0, cols - 1)
    r2 = random.randint(r1, rows - 1)
    c2 = random.randint(c1, cols - 1)
    return ((r1, c1), (r2, c2))

def generate_action_list(grid_width=17, grid_height=10, n_actions=20):
    action_list = []

    for _ in range(n_actions):
        # 시작 col, row 선택
        start_col = random.randint(0, grid_width - 1)
        start_row = random.randint(0, grid_height - 1)

        # 2~4 cell 포함하는 작은 영역 만들기
        # width, height를 랜덤하게 선택
        width = random.choice([0, 1, 2, 3, 4])
        height = random.choice([0, 1, 2, 3, 4])

        # 끝 col, row를 계산 (grid 넘어가면 최대값 clamp)
        end_col = min(start_col + width, grid_width - 1)
        end_row = min(start_row + height, grid_height - 1)

        action = ((start_col, start_row), (end_col, end_row))
        action_list.append(action)

    return action_list

def test_dynamics_handler_rollout(model_path="cnn_model_weights.pth", rollout_len=10):
    # 0. 액션 리스트 불러오기
    action_list = generate_action_list(n_actions=rollout_len)

    # 1. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDigitClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 핸들러 생성
    handler = DynamicsHandler(model=model, grid_shape=(17, 10))

    # 3. 초기 observation
    obs = handler.get_observation()

    x0, y0 = handler.top_left_cell_coordinate
    x1, y1 = handler.bottom_right_cell_coordinate

    print(f"Top-left cell coordinate: ({x0}, {y0})")
    print(f"Bottom-right cell coordinate: ({x1}, {y1})")

    pyautogui.moveTo(x0, y0)
    # time.sleep(1)  # 1초 쉬면서 위치 확인

    pyautogui.moveTo(x1, y1)
    # time.sleep(1)  # 1초 쉬면서 위치 확인

    trajectory = []

    for step in range(rollout_len):
        # action = generate_random_action(handler.grid_shape)
        action = action_list[step]
        next_obs, reward, done = handler.step(action, step)

        print(f"[Step {step}] Action: {action}, Reward: {reward}, Done: {done}")
        trajectory.append((obs, action, reward, next_obs, done))
        obs = next_obs

        if done:
            break

    print(f"\nTrajectory length: {len(trajectory)} steps")
    return trajectory

if __name__ == "__main__":
    trajectory = test_dynamics_handler_rollout()
