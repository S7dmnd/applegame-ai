import pyautogui
import cv2
import numpy as np
from PIL import Image
from scripts.utils.image_differentiator import ImageDifferentiator
from scripts.utils import env_utils as eu
from scripts.utils import pytorch_utils as ptu
import time

class DynamicsHandler:
    def __init__(self, model, grid_shape=(17, 10), margin_w_ratio=0.046, margin_h_ratio=0.039, max_episode_size=150):
        self.model = model  # Differentiator용 CNN
        self.differentiator = None # 구조 수정 완료; 처음에만 instanciation함
        self.crop_ratio = (0.053, 0.08, 0.928, 0.94) #이것도 나중에는 구조 수정하면 위의 self.differentiator에서 가져올거임
        self.grid_shape = grid_shape # (가로, 세로)
        self.current_grid = None
        self.current_score = 0
        self.previous_score = 0
        self.step_count = 0

        self.done = False
        self.max_episode_size = max_episode_size

        self.game_coordinates = () # 게임의 absolute (Top-left, Bottom-right) pixel 위치 저장
        self.top_left_cell_coordinate = () # 제일 왼쪽 위 셀의 왼쪽 위 코너 좌표
        self.bottom_right_cell_coordinate = () # 제일 오른쪽 아래 셀의 오른쪽 아래 코너 좌표
        self.margin_w_ratio = margin_w_ratio
        self.margin_h_ratio = margin_h_ratio
        self.w_per_cell = 0
        self.h_per_cell = 0
        
        # 스크린 해상도 기준 - 필요하면 calibrate
        self.screen_width, self.screen_height = pyautogui.size()

    def reset(self):
        self.current_grid = None
        self.previous_score = 0
        self.current_score = 0
        self.done = False
        self.step_count = 0
        return self.get_observation()

    def autocrop_game_area_cv(self, pil_image):
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([74, 250, 200])
        upper_green = np.array([76, 255, 209])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 바깥에서 margin_ratio 만큼 안쪽으로 들어가서 crop
        margin_w = int(w * self.margin_w_ratio)
        margin_h = int(h * self.margin_h_ratio)

        x_new = x + margin_w
        y_new = y + margin_h
        w_new = w - 2 * margin_w
        h_new = h - 2 * margin_h

        self.game_coordinates = ((x_new, y_new), (x_new + w_new, y_new + h_new))
        crop_left, crop_top, crop_right, crop_bottom = self.crop_ratio
        final_x0 = int(x_new + crop_left * w_new)
        final_y0 = int(y_new + crop_top * h_new)
        final_x1 = int(x_new + crop_right * w_new)
        final_y1 = int(y_new + crop_bottom * h_new)
        self.top_left_cell_coordinate = (final_x0, final_y0)
        self.bottom_right_cell_coordinate = (final_x1, final_y1)
        self.w_per_cell = (final_x1 - final_x0) / self.grid_shape[0]
        self.h_per_cell = (final_y1 - final_y0) / self.grid_shape[1]

        cropped = cv_image[y_new:y_new+h_new, x_new:x_new+w_new]

        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return cropped_pil


    def capture_screen(self):
        """현재 화면 스크린샷 찍고, 게임 화면만 autocrop"""
        screenshot = pyautogui.screenshot()
        cropped_game_area = self.autocrop_game_area_cv(screenshot)
        return cropped_game_area

    def get_observation(self):
        """현재 게임판 상태를 2D Grid로 변환 (Observation)"""
        image = self.capture_screen()

        # Differentiator 구조 바꿔서 초기화/이미지 갈아끼기 했음
        if self.differentiator:
            self.differentiator.reset(image)
        else:
            self.differentiator = ImageDifferentiator(model=self.model, image=image, grid_shape=self.grid_shape) 
        self.differentiator.preprocess_crop()
        grid = self.differentiator.differentiate()
        self.current_grid = grid
        self.previous_score = self.current_score
        self.current_score = (self.current_grid == 0).sum().sum()
        # print(self.current_grid)

        return eu.to_tensor(grid=self.current_grid, device=ptu.device)

    def perform_action(self, action):
        """
        action: ((start_col, start_row), (end_col, end_row))
        두 셀 좌표 받아서 드래그 액션 수행
        """
        col1, row1, col2, row2 = np.floor(action).astype(int)
        start_x = round(self.top_left_cell_coordinate[0] + col1 * self.w_per_cell)
        start_y = round(self.top_left_cell_coordinate[1] + row1 * self.h_per_cell)
        end_x = round(self.top_left_cell_coordinate[0] + (col2 + 1) * self.w_per_cell)
        end_y = round(self.top_left_cell_coordinate[1] + (row2 + 1) * self.h_per_cell)

        # 마우스 누르고
        pyautogui.moveTo(start_x, start_y)
        pyautogui.mouseDown()

        # 드래그
        pyautogui.moveTo(end_x, end_y, duration=0.3, tween=pyautogui.easeInOutQuad)

        # 약간 대기 (예: 0.1초)
        time.sleep(0.2)

        # 마우스 떼기
        pyautogui.mouseUp()


    def step(self, action):
        """Agent가 action을 주면, 환경을 한 스텝 진행시킨다"""
        self.perform_action(action)

        # 드래그 후 잠깐 기다려야 게임판이 업데이트됨
        pyautogui.sleep(0.7)

        next_grid = self.get_observation()
        reward = self.current_score - self.previous_score # "소요시간" 까지 고려해서, 무의미한 액션에는 페널티
        self.step_count += 1
        done = self.check_done(self.step_count)

        return next_grid, reward, done


    def check_done(self, step):
        """게임이 끝났는지 확인"""
        # 사과 다 없어진 경우 끝이라고 볼 수 있음
        # 이거 전용 CNN 또 만들어야 될 듯 (씨발)
        # 오래 걸릴 것 같진 않음 binary classification이라서...
        # 원래는 image CNN돌려야되는데 일단은 trajectory 크기를 받아서 끝내기로 
        # 나중에는 이걸 get_observation에서 스샷떠서 받은 image를 CNN에 돌려서 판단하기로.

        return self.max_episode_size <= step

