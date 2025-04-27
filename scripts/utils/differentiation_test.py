# test_image_differentiator.py

from PIL import Image
# import torch
from image_differentiator import ImageDifferentiator  # 모듈화된 클래스 불러오기

# 이미지 파일 경로
image_path = "data/1.png"  # 적절한 경로로 변경해야 합니다.
image = Image.open(image_path)  # 테스트할 이미지 파일 열기

# 모델과 ImageDifferentiator 객체 생성
dummy_model = None
differentiator = ImageDifferentiator(dummy_model, image)

# 테스트 1: preprocess_crop (이미지 크롭 테스트)
differentiator.preprocess_crop(left_percent=0.105, top_percent=0.08, right_percent=0.926, bottom_percent=0.94)
differentiator.image.show()  # 크롭된 이미지 확인

import matplotlib.pyplot as plt

# 테스트 2: split_into_grids (그리드 분할 테스트)
grids = differentiator.split_into_grids()

# 이미지 시각화 (가로 10, 세로 17 그리드로 출력)
fig, axes = plt.subplots(differentiator.grid_shape[1], differentiator.grid_shape[0], figsize=(12, 10))

for row in range(differentiator.grid_shape[1]):
    for col in range(differentiator.grid_shape[0]):
        cell_image = grids.iloc[row, col]  # 셀 이미지
        axes[row, col].imshow(cell_image)  # 해당 셀 이미지를 subplot에 표시
        axes[row, col].axis('off')  # 축 제거

plt.show()  # 전체 이미지를 한 번에 출력
