import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from simple_digit_classifier import SimpleDigitClassifier
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn

def train_cnn_model(save_path="cnn_model_weights.pth", epochs=5, batch_size=64, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로더 (MNIST or 너가 커스텀한 데이터)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = ImageFolder(root='./data/cell_images', transform=transform)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실함수, 최적화기
    model = SimpleDigitClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습 루프
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation Accuracy 체크
    # model.validate_model(dataloader=val_loader, device=device)
    model.evaluate_accuracy(dataloader=val_loader, device=device)

    # 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    return model

if __name__=="__main__":
    for i in range(10):
        print(f"Test {i}")
        train_cnn_model()