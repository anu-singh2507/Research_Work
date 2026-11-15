import torch
import numpy as np
from torch.utils.data import DataLoader
from multimodal_dataset import get_dataloaders, TOTAL_HPO_FEATURES
from fusion_model import CrossModalFusionModel

def evaluate_ranking_metrics(model, data_loader, device):
    model.eval()
    all_ranks = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, phenotypes, labels in data_loader:
            images = images.to(device)
            phenotypes = phenotypes.to(device)
            labels = labels.to(device)

            logits, _, _, _ = model(images, phenotypes)
            probs = torch.softmax(logits, dim=1)

            # Sort predictions descending for ranking metrics
            sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)

            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_ranking = sorted_indices[i].tolist()

                # Find rank of true label (1-indexed)
                rank = pred_ranking.index(true_label) + 1
                all_ranks.append(1.0 / rank)

                if rank == 1:
                    top1_correct += 1
                if rank <= 3:
                    top3_correct += 1
                if rank <= 5:
                    top5_correct += 1
                total_samples += 1

    top1_recall = top1_correct / total_samples
    top3_recall = top3_correct / total_samples
    top5_recall = top5_correct / total_samples
    mrr = np.mean(all_ranks)

    return top1_recall, top3_recall, top5_recall, mrr

if __name__ == "__main__":
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    DATASET_PATH = r"D:\anu_research"
    _, val_loader = get_dataloaders(DATASET_PATH, batch_size=8)

    model = CrossModalFusionModel(num_hpo_features=TOTAL_HPO_FEATURES, embedding_dim=512, num_classes=5)
    model.load_state_dict(torch.load("best_rare_disease_model.pth", map_location=device))
    model.to(device)

    t1, t3, t5, mrr = evaluate_ranking_metrics(model, val_loader, device)
    print("\n" + "="*45)
    print("   FORMAL THESIS EVALUATION METRICS REPORT   ")
    print("="*45)
    print(f"  - Top-1 Recall: {t1 * 100:.2f}%")
    print(f"  - Top-3 Recall: {t3 * 100:.2f}%")
    print(f"  - Top-5 Recall: {t5 * 100:.2f}%")
    print(f"  - Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("="*45)