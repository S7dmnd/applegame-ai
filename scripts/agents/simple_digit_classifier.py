import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch

class SimpleDigitClassifier(nn.Module):
    def __init__(self):
        super(SimpleDigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # (1, 64, 64) -> (32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # (32, 64, 64) -> (64, 64, 64)
        self.pool = nn.MaxPool2d(2, 2)  # (64, 64, 64) -> (64, 16, 16)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)  # Output 10 classes (1~9 숫자 + empty(0))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def validate_model(self, dataloader, device, n_images=32):
        """Validation set 이미지 + 예측 + 정답 + Accuracy 출력"""
        self.eval()
        images_shown = 0
        correct = 0
        total = 0

        fig = plt.figure(figsize=(12, 8))
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                batch_size = inputs.size(0)

                for idx in range(batch_size):
                    if images_shown >= n_images:
                        break

                    ax = plt.subplot(4, 8, images_shown + 1)
                    img = inputs[idx].cpu().permute(1, 2, 0)
                    ax.imshow(img, cmap="gray" if img.shape[2] == 1 else None)
                    ax.set_title(f"P:{preds[idx].item()} / T:{labels[idx].item()}")
                    ax.axis('off')

                    images_shown += 1

                if images_shown >= n_images:
                    break

        plt.tight_layout()
        plt.show()

        # Accuracy 계산해서 출력
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

        return accuracy

    def evaluate_accuracy(self, dataloader, device):
        """Validation/Test 세트에 대한 Accuracy만 계산해서 리턴"""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy
