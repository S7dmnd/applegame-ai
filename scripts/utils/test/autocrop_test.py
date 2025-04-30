# test_autocrop.py

from PIL import Image
import pyautogui
import time
from dynamics_handler import DynamicsHandler
from scripts.agents.simple_digit_classifier import SimpleDigitClassifier  # (dummy model이라도 필요)
import torch

def main():
    # dummy model
    dummy_model = SimpleDigitClassifier()
    handler = DynamicsHandler(model=dummy_model, margin_w_ratio=0.046, margin_h_ratio=0.039)

    cropped_img = handler.capture_screen()

    x0, y0 = handler.top_left_cell_coordinate
    x1, y1 = handler.bottom_right_cell_coordinate

    print(f"Top-left cell coordinate: ({x0}, {y0})")
    print(f"Bottom-right cell coordinate: ({x1}, {y1})")

    #pyautogui.moveTo(x0, y0)
    #time.sleep(1)  # 1초 쉬면서 위치 확인

    #pyautogui.moveTo(x1, y1)
    #time.sleep(1)  # 1초 쉬면서 위치 확인

    handler.perform_action(((16, 2), (16, 3)))

    # cropped_img.show()  # 화면에 바로 보여주기
    # cropped_img.save("cropped_result.png")  # 파일로 저장하고 싶으면



if __name__ == "__main__":
    main()
