import os
import torch
import torch.nn as nn
import torch.optim as optim
from multimodal_dataset import get_dataloaders, TOTAL_HPO_FEATURES
from fusion_model import CrossModalFusionModel

def train_framework():
    # 1. Setup Hardware (AMD GPU via DirectML)
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"\n[Hardware] Training on AMD GPU via DirectML: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Hardware] DirectML fallback device: {device}")

    # 2. Hyperparameters for Few-Shot / Multimodal Framework
    DATASET_PATH = r"D:\anu_research"
    batch_size = 8
    epochs = 5  # Increase epochs later for full convergence
    learning_rate = 1e-4

    # 3. Load Data
    print("[Data] Initializing multimodal data loaders for 5 rare diseases...")
    train_loader, val_loader = get_dataloaders(DATASET_PATH, batch_size=batch_size, train_split=0.8)

    # 4. Initialize Model, Loss Function, and Optimizer
    model = CrossModalFusionModel(num_hpo_features=TOTAL_HPO_FEATURES, embedding_dim=512, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\n[Training Started] Total Epochs: {epochs} | Batch Size: {batch_size}")

    # 5. Training & Validation Loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, phenotypes, labels in train_loader:
            images = images.to(device)
            phenotypes = phenotypes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass through Cross-Modal Fusion Network
            logits, z_img, z_graph, z_fused = model(images, phenotypes)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass & optimization step
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train

        # Validation Step
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, phenotypes, labels in val_loader:
                images = images.to(device)
                phenotypes = phenotypes.to(device)
                labels = labels.to(device)

                logits, _, _, _ = model(images, phenotypes)
                _, predicted = torch.max(logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val if total_val > 0 else 0.0

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Accuracy: {epoch_acc * 100:.2f}% | "
              f"Validation Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_rare_disease_model.pth")

    print(f"\n[Complete] Training finished successfully! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("Model weights saved to 'best_rare_disease_model.pth'.")

if __name__ == "__main__":
    train_framework()