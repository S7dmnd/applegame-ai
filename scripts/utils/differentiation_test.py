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
differentiator.preprocess_crop(left_percent=0.105, top_percent=0.08, right_percent=0.926, bottom_percent=0.93)
differentiator.image.show()  # 크롭된 이미지 확인

# 테스트 2: split_into_grids (그리드 분할 테스트)
grids = differentiator.split_into_grids()
# 각 그리드 셀 이미지 확인
for row in range(grids.shape[0]):  # 세로
    for col in range(grids.shape[1]):  # 가로
        cell_image = grids.iloc[row, col]  # 셀 이미지
        cell_image.show()  # 셀 이미지를 팝업으로 보여줌