import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import h5py
from torchvision import transforms
from PIL import Image
from efficientnet_b0_m1 import EnhancedEfficientNetB0
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    return accuracy, recall, f1


def plot_metrics(accuracy, recall, f1):
    metrics = ['Accuracy', 'Recall', 'F1 Score']
    values = [accuracy, recall, f1]

    plt.figure(figsize=(10, 6))
    bar_width = 0.5
    bar_colors = [(106 / 255, 128 / 255, 185 / 255),
                  (246 / 255, 199 / 255, 148 / 255),
                  (255 / 255, 246 / 255, 179 / 255)]
    bars = plt.bar(metrics, values, width=bar_width, color=bar_colors)

    # 红色参考线
    plt.axhline(y=1, color='red', linestyle='--', linewidth=5)

    # 添加柱状图数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.4f}',
                 ha='center', va='bottom',
                 fontsize=12)

    # 添加差值显示
    for bar in bars:
        y_top = bar.get_height()
        delta = 1 - y_top

        if delta > 0:
            # 计算中间位置
            mid_x = bar.get_x() + bar.get_width() / 2
            mid_y = (y_top + 1) / 2

            # 添加差值文本
            plt.text(mid_x + 0.1,  # 向右偏移
                     mid_y,
                     f'Δ = {delta:.4f}',
                     ha='left', va='center',
                     fontsize=12,
                     color='darkred',
                     bbox=dict(boxstyle='round',
                               facecolor='white',
                               edgecolor='lightgray',
                               alpha=0.8))

    plt.ylim(0.99, 1.0)
    plt.title('Concrete Crack Detection Evaluation', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def main_evaluation():
    # Configuration
    file_path = r'Your data set file location.h5'
    batch_size = 32
    img_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    with h5py.File(file_path, 'r') as hf:
        X = np.array(hf.get('X_concrete'))
        y = np.array(hf.get("y_concrete"))

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Convert data to tensors
    X_transformed = []
    for image in X:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1:
            image = image / 255.0

        if len(image.shape) == 2:
            image = image[..., np.newaxis]

        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        image_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(image_tensor)
        X_transformed.append(image_tensor)

    X_tensor = torch.stack(X_transformed)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create full dataset
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # Calculate sizes for random split (20% for test)
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size

    # Randomly split the dataset
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = EnhancedEfficientNetB0(num_classes=2).to(device)

    # Load saved weights
    checkpoint = torch.load('final_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate model
    accuracy, recall, f1 = evaluate_model(model, test_loader, device)

    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot metrics
    plot_metrics(accuracy, recall, f1)


if __name__ == '__main__':
    main_evaluation()
