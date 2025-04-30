import torch
from PIL import Image
from scripts.agents.simple_digit_classifier import SimpleDigitClassifier
from image_differentiator import ImageDifferentiator

def test_differentiator(model_path="cnn_model_weights.pth", test_image_path="test.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 불러오기
    model = SimpleDigitClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("모델 로딩 완료")

    # 2. 테스트 이미지 불러오기
    image = Image.open(test_image_path)
    print("테스트 이미지 로딩 완료")

    # 3. Differentiator 인스턴스 생성
    differentiator = ImageDifferentiator(model=model, image=image, grid_shape=(17, 10))  # 게임에 맞춰 17x10
    print("Differentiator 인스턴스 생성 완료")

    # 4. (선택) 크롭 진행
    differentiator.preprocess_crop()
    differentiator.image.show()
    print("이미지 crop 완료")

    # 5. 그리드로 분할
    grids = differentiator.split_into_grids()
    differentiator.grids_visualization()
    print(f"그리드 분할 완료: shape={grids.shape}")

    # 6. 숫자 예측
    number_grid = differentiator.differentiate()
    print("숫자 인식 완료")

    # 7. 결과 출력
    print("예측된 2D 그리드:")
    print(number_grid)

    return number_grid


if __name__ == "__main__":
    test_differentiator(
        model_path="cnn_model_weights.pth",
        test_image_path="cropped_result.png"
    )