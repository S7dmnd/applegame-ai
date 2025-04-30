# test_image_differentiator.py

from PIL import Image
# import torch

from image_differentiator import ImageDifferentiator  # 모듈화된 클래스 불러오기

#for i in range (1, 7):

i = "test"

# 이미지 파일 경로
image_path = f"data/{i}.png"  # 적절한 경로로 변경해야 합니다.
print(f"{image_path} 테스팅 시작")
image = Image.open(image_path)  # 테스트할 이미지 파일 열기

# 모델과 ImageDifferentiator 객체 생성
dummy_model = None
differentiator = ImageDifferentiator(dummy_model, image)

# 테스트 1: preprocess_crop (이미지 크롭 테스트)
differentiator.preprocess_crop()
differentiator.image.show()  # 크롭된 이미지 확인

import matplotlib.pyplot as plt

# 테스트 2: split_into_grids (그리드 분할 테스트)
grids = differentiator.split_into_grids()

differentiator.grids_visualization()
#differentiator.save_cell_images(prefix=i)