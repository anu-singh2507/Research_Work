import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from multimodal_dataset import RareDiseaseMultimodalDataset, TOTAL_HPO_FEATURES
from fusion_model import CrossModalFusionModel

def run_few_shot_experiment(n_shots_per_class):
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    DATASET_PATH = r"D:\anu_research"
    full_dataset = RareDiseaseMultimodalDataset(data_root_dir=DATASET_PATH, transform=None)

    # Group sample indices by their class label
    class_indices = {i: [] for i in range(5)}
    for idx in range(len(full_dataset)):
        _, _, label = full_dataset[idx]
        class_indices[label.item()].append(idx)

    # Select exactly n_shots_per_class for each class
    selected_indices = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        selected_indices.extend(indices[:n_shots_per_class])

    few_shot_subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(few_shot_subset, batch_size=min(4, len(few_shot_subset)), shuffle=False)

    # Load Model
    model = CrossModalFusionModel(num_hpo_features=TOTAL_HPO_FEATURES, embedding_dim=512, num_classes=5)
    model.load_state_dict(torch.load("best_rare_disease_model.pth", map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, phenotypes, labels in loader:
            images = images.to(device)
            phenotypes = phenotypes.to(device)
            labels = labels.to(device)

            logits, _, _, _ = model(images, phenotypes)
            _, predicted = torch.max(logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy

if __name__ == "__main__":
    print("\n" + "="*50)
    print("   FEW-SHOT ROBUSTNESS EVALUATION (n-shots)   ")
    print("="*50)
    
    shot_sizes = [1, 5, 10, 20]
    for n in shot_sizes:
        acc = run_few_shot_experiment(n_shots_per_class=n)
        print(f"  - Few-Shot Size (n = {n} scans/class) | Accuracy: {acc:.2f}%")
    
    print("="*50)