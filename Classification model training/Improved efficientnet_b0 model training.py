import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# Advanced ECA Attention Module
class AdvancedECAAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(AdvancedECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = y_avg + y_max
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        x = x * y.expand_as(x)

        # Spatial attention
        spatial = torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1)
        spatial = self.sigmoid(self.spatial_conv(spatial))
        return x * spatial


# Improved MBConv Block
class ImprovedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1, reduction_ratio=4):
        super(ImprovedMBConvBlock, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expansion_factor

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expansion_factor != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        self.attention = AdvancedECAAttention(hidden_dim)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip = stride == 1 and in_channels == out_channels
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.attention(x)
        x = self.project(x)
        x = self.dropout(x)

        if self.skip:
            x = x + identity
        return x


# Enhanced EfficientNetB0
class EnhancedEfficientNetB0(nn.Module):
    def __init__(self, num_classes=2):
        super(EnhancedEfficientNetB0, self).__init__()

        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            ImprovedMBConvBlock(32, 64, expansion_factor=4),
            ImprovedMBConvBlock(64, 128, expansion_factor=4, stride=2),
            ImprovedMBConvBlock(128, 256, expansion_factor=4, stride=2)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        avg_pool = self.global_pool[0](x).squeeze(-1).squeeze(-1)
        max_pool = self.global_pool[1](x).squeeze(-1).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.classifier(x)
        return x


# Data loading and preprocessing
def load_and_preprocess_data(file_path):
    print("Loading the data...")
    with h5py.File(file_path, 'r') as hf:
        X = np.array(hf.get('X_concrete'))
        y = np.array(hf.get("y_concrete"))
    print("Data successfully loaded!")

    print("Scaling the data...")
    X = X.astype(np.float32)
    X = X / 255.0
    print("Data successfully scaled!")

    return X, y


# Advanced data augmentation
def get_advanced_transforms(img_size=128):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, shear=10),
            transforms.RandomPerspective(0.2)
        ], p=0.7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# Image conversion utility
def convert_to_pil(image):
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return Image.fromarray(image)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    training_accuracy = []
    validation_accuracy = []
    training_loss = []
    validation_loss = []
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        training_accuracy.append(train_accuracy)
        training_loss.append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * correct / total
        validation_accuracy.append(val_accuracy)
        validation_loss.append(val_loss / len(val_loader))

        # Print epoch results
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Training Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print("Validation loss improved - saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

        if scheduler:
            scheduler.step()

    return training_accuracy, validation_accuracy, training_loss, validation_loss


# Visualization function
# [Previous imports and model classes remain the same until the visualization function]

# Enhanced curve smoothing function
def smooth_curve(x, y, points=200):
    """
    Creates a smoother curve with more points and better interpolation
    """
    x_new = np.linspace(min(x), max(x), points)
    try:
        spl = make_interp_spline(x, y, k=3)  # Use cubic spline
        y_smooth = spl(x_new)
        # Ensure no negative values for loss
        if 'loss' in str(y).lower():
            y_smooth = np.maximum(y_smooth, 0)
        return x_new, y_smooth
    except:
        # Fallback to linear interpolation if spline fails
        return x, y


# Enhanced visualization function
def plot_training_results(training_accuracy, validation_accuracy, training_loss, validation_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Convert lists to numpy arrays
    epochs = np.arange(1, len(training_accuracy) + 1)

    # Smooth curves
    x_smooth, train_acc_smooth = smooth_curve(epochs, np.array(training_accuracy))
    _, val_acc_smooth = smooth_curve(epochs, np.array(validation_accuracy))
    x_smooth_loss, train_loss_smooth = smooth_curve(epochs, np.array(training_loss))
    _, val_loss_smooth = smooth_curve(epochs, np.array(validation_loss))

    # Plot accuracy
    ax1.plot(x_smooth, train_acc_smooth, label='Training Accuracy', linewidth=2)
    ax1.plot(x_smooth, val_acc_smooth, label='Validation Accuracy', linewidth=2, linestyle='--')
    ax1.set_title('Model Accuracy Over Time', fontsize=12, pad=15)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1, len(training_accuracy))

    # Plot loss
    ax2.plot(x_smooth_loss, train_loss_smooth, label='Training Loss', linewidth=2)
    ax2.plot(x_smooth_loss, val_loss_smooth, label='Validation Loss', linewidth=2, linestyle='--')
    ax2.set_title('Model Loss Over Time', fontsize=12, pad=15)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.set_xlim(1, len(training_loss))

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Configuration
    file_path = r'Your data set file location.h5'
    batch_size = 32
    num_epochs = 10
    img_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare data transforms
    transform = get_advanced_transforms(img_size)

    print("Converting images to tensors...")
    # Convert data to PyTorch format with progress tracking
    X_train_transformed = []
    X_val_transformed = []

    for i, image in enumerate(X_train):
        if i % 1000 == 0:
            print(f"Processing training image {i}/{len(X_train)}")
        X_train_transformed.append(transform(convert_to_pil(image)))

    for i, image in enumerate(X_val):
        if i % 1000 == 0:
            print(f"Processing validation image {i}/{len(X_val)}")
        X_val_transformed.append(transform(convert_to_pil(image)))

    X_train_transformed = torch.stack(X_train_transformed)
    X_val_transformed = torch.stack(X_val_transformed)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_transformed, y_train)
    val_dataset = TensorDataset(X_val_transformed, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and training components
    model = EnhancedEfficientNetB0(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    print("Starting training...")
    # Train model
    training_accuracy, validation_accuracy, training_loss, validation_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
    )

    print("Training completed. Plotting results...")
    # Plot results with smoothed curves
    plot_training_results(training_accuracy, validation_accuracy, training_loss, validation_loss)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy,
        'training_loss': training_loss,
        'validation_loss': validation_loss
    }, 'final_model.pth')

    print("Results saved. Training complete!")


if __name__ == '__main__':
    main()