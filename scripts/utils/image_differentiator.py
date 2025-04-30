import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
from datetime import datetime

class ImageDifferentiator:
    def __init__(self, model, image: Image, grid_shape=(17, 10), crop_ratio=(0.053, 0.08, 0.928, 0.94)):
        self.model = model  # CNN 모델
        self.grid_shape = grid_shape  # (가로, 세로) 형태
        self.image = image
        # 빈 DataFrame 생성, grid_shape에 맞게 크기 지정
        self.grids = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))
        self.number_grid = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))  # 숫자 그리드 저장
        (self.left_percent, self.top_percent, self.right_percent, self.bottom_percent) = crop_ratio
    
    def preprocess_crop(self):
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
        left = int(self.left_percent * w)
        top = int(self.top_percent * h)
        right = int(self.right_percent * w)
        bottom = int(self.bottom_percent * h)

        # 이미지 크롭
        self.image = self.image.crop((left, top, right, bottom))
        return self.image

    def split_into_grids(self):
        """게임 화면 이미지를 grid_shape에 맞게 분할"""
        w, h = self.image.size
        grid_w = w / self.grid_shape[0]  # 정수 나누기 X, 실수 나누기
        grid_h = h / self.grid_shape[1]

        for row in range(self.grid_shape[1]):
            for col in range(self.grid_shape[0]):
                left = int(col * grid_w)
                top = int(row * grid_h)

                # 마지막 열 / 마지막 행은 이미지 경계까지
                if col == self.grid_shape[0] - 1:
                    right = w
                else:
                    right = int((col + 1) * grid_w)

                if row == self.grid_shape[1] - 1:
                    bottom = h
                else:
                    bottom = int((row + 1) * grid_h)

                cell = self.image.crop((left, top, right, bottom))
                self.grids.iloc[row, col] = cell

        return self.grids

    def save_cell_images(self, prefix="", save_dir="./data/cell_images"):
        """self.grids의 셀 이미지를 각각 파일로 저장"""
        # 저장할 디렉토리 만들기
        os.makedirs(save_dir, exist_ok=True)

        # 현재 시간 가져오기
        now = datetime.now()
        time_prefix = now.strftime("%m%d%H%M")  # MMDDHHMM

        for row in range(self.grid_shape[1]):
            for col in range(self.grid_shape[0]):
                cell_img = self.grids.iloc[row, col]
                filename = f"{prefix}-{time_prefix}-{row}-{col}.png"
                save_path = os.path.join(save_dir, filename)
                cell_img.save(save_path)

        print(f"Cell images saved to {save_dir}")

    def grids_visualization(self):
        fig, axes = plt.subplots(self.grid_shape[1], self.grid_shape[0], figsize=(12, 10))

        for row in range(self.grid_shape[1]):
            for col in range(self.grid_shape[0]):
                cell_image = self.grids.iloc[row, col]  # 셀 이미지
                axes[row, col].imshow(cell_image)  # 해당 셀 이미지를 subplot에 표시
                axes[row, col].axis('off')  # 축 제거

        plt.show()  # 전체 이미지를 한 번에 출력

    def predict_grid(self, cell_img):
        """한 칸 이미지에 대해 숫자 예측"""
        tensor = self.preprocess_cells(cell_img)  # 전처리된 텐서
        with torch.no_grad():
            pred = self.model(tensor.unsqueeze(0))  # (1, C, H, W) 형태로 모델에 입력
            pred_label = pred.argmax(dim=1).item()  # 예측된 클래스(숫자)
        return pred_label

    def preprocess_cells(self, cell_img):
        """셀 이미지를 모델 입력에 맞게 전처리"""

        if cell_img.mode != "RGB":
            cell_img = cell_img.convert("RGB")
        
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        tensor = transform(cell_img)  # (3, 64, 64) tensor로 변환 + normalize (-1 ~ 1)
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)
        return tensor

    def differentiate(self):
        """전체 이미지에서 2D 배열 숫자 결과 리턴"""
        grids = self.split_into_grids()  # 이미지를 그리드로 나누기
        number_grid = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))  # DataFrame으로 바꿔!

        for row in range(self.grid_shape[1]):
            for col in range(self.grid_shape[0]):
                cell = grids.iloc[row, col]  # (row, col) 순서
                number = self.predict_grid(cell)
                number_grid.iloc[row, col] = number

        self.number_grid = number_grid
        return number_grid

    def reset(self, image):
        self.image = image
        self.grids = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))
        self.number_grid = pd.DataFrame(index=range(self.grid_shape[1]), columns=range(self.grid_shape[0]))