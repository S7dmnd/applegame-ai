import pandas as pd
import numpy as np
from scripts.utils.virtual_dynamics_handler import VirtualDynamicsHandler

def test_perform_action_variants():
    handler = VirtualDynamicsHandler(seed=1)
    handler.reset()  # 초기화: 랜덤한 grid 생성

    # 테스트용으로 합이 10이 되도록 몇 개 고정 설정 (위치를 명확하게 하기 위함)
    handler.current_grid.iloc[0, 0] = 4
    handler.current_grid.iloc[0, 1] = 6  # (col=0,row=0)~(col=1,row=0) → 합: 10
    handler.current_grid.iloc[1, 0] = 3
    handler.current_grid.iloc[2, 0] = 7  # 수직 합 10
    handler.current_grid.iloc[2, 1] = 2
    handler.current_grid.iloc[2, 2] = 8  # 대각선
    handler.current_grid.iloc[1, 2] = 5
    handler.current_grid.iloc[0, 2] = 5  # 역방향 대각선

    print("📦 초기 Grid:")
    print(handler.current_grid)

    print("\n⚡ Action 1: 오른쪽 아래로 드래그")
    handler.step(((0, 0), (1, 0)))  # 수평
    print(handler.current_grid)

    print("\n⚡ Action 2: 왼쪽 아래로 드래그")
    handler.step(((0, 3), (0, 0)))  # 수직 역방향
    print(handler.current_grid)

    print("\n⚡ Action 3: 오른쪽 위로 드래그")
    handler.step(((1, 3), (2, 1)))  # 수직 정방향
    print(handler.current_grid)

    print("\n⚡ Action 4: 왼쪽 위로 드래그")
    handler.step(((5, 7), (2, 5)))  # 대각 역방향
    print(handler.current_grid)

    print("\n⚡ Action 5: 여러 수 Popping")
    _, reward, _ = handler.step(((15, 8), (16, 9)))
    print(handler.current_grid)
    print(f"current score: {handler.current_score}, reward: {reward}")

if __name__ == "__main__":
    test_perform_action_variants()
