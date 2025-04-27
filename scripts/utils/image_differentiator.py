import numpy as np
import pandas as pd
from PIL import Image
# import torch

class ImageDifferentiator:
    def __init__(self, model, image: Image, grid_shape=(16, 10)):
        self.model = model  # CNN 모델
        self.grid_shape = grid_shape  # (세로, 가로) 형태
        self.image = image
        # 빈 DataFrame 생성, grid_shape에 맞게 크기 지정
        self.grids = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))
        self.number_grid = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))  # 숫자 그리드 저장
    
    def preprocess_crop(self, left_percent: float, top_percent: float, right_percent: float, bottom_percent: float):
        """
        왼쪽, 위, 오른쪽, 아래 값을 이미지 크기에 대한 비율로 받아서 크롭한다.
        :param left_percent: 이미지 왼쪽에서 크롭할 비율 (0~1 사이)
        :param top_percent: 이미지 위쪽에서 크롭할 비율 (0~1 사이)
        :param right_percent: 이미지 오른쪽에서 크롭할 비율 (0~1 사이)
        :param bottom_percent: 이미지 아래쪽에서 크롭할 비율 (0~1 사이)
        """
        # 이미지 크기 얻기
        w, h = self.image.size

        # 좌표 계산 (비율 * 이미지 크기)
        left = int(left_percent * w)
        top = int(top_percent * h)
        right = int(right_percent * w)
        bottom = int(bottom_percent * h)

        # 이미지 크롭
        self.image = self.image.crop((left, top, right, bottom))
        return self.image


    def split_into_grids(self):
        """게임 화면 이미지를 grid_shape에 맞게 분할"""
        w, h = self.image.size
        grid_w = w // self.grid_shape[0]
        grid_h = h // self.grid_shape[1]

        for row in range(self.grid_shape[1]):
            for col in range(self.grid_shape[0]):
                left = col * grid_w
                top = row * grid_h
                right = left + grid_w
                bottom = top + grid_h
                cell = self.image.crop((left, top, right, bottom))
                self.grids.iloc[row, col] = cell  # grid_data를 DataFrame에 바로 저장

        return self.grids  # 2D DataFrame (그리드들)

    def predict_grid(self, cell_img):
        """한 칸 이미지에 대해 숫자 예측"""
        tensor = self.preprocess_cells(cell_img)  # 전처리된 텐서
        with torch.no_grad():
            pred = self.model(tensor.unsqueeze(0))  # (1, C, H, W) 형태로 모델에 입력
            pred_label = pred.argmax(dim=1).item()  # 예측된 클래스(숫자)
        return pred_label

    def preprocess_cells(self, cell_img):
        """셀 이미지를 모델 입력에 맞게 전처리"""
        # 이미지 리사이즈, 정규화, 텐서 변환 (예: 64x64로 리사이즈)
        cell_img = cell_img.resize((64, 64))  # 예시로 64x64 크기 조정
        tensor = torch.tensor(np.array(cell_img) / 255.0).float()  # 정규화
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64) 형태로 변환
        return tensor

    def differentiate(self):
        """전체 이미지에서 2D 배열 숫자 결과 리턴"""
        grids = self.split_into_grids()  # 이미지를 그리드로 나누기
        number_grid = np.empty(self.grid_shape, dtype=int)  # 빈 numpy 배열로 숫자 그리드 초기화

        for row in range(self.grid_shape[1]):
            for col in range(self.grid_shape[0]):
                cell = grids.iloc[row, col]  # 해당 위치의 셀 이미지
                number_grid[row, col] = self.predict_grid(cell)  # 각 셀 예측 후 저장

        self.number_grid = pd.DataFrame(number_grid)  # 숫자 그리드를 DataFrame으로 저장
        return self.number_grid  # 숫자 그리드 결과
