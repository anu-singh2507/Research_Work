import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity (MS) Loss for Deep Metric Learning.
    Weights positive and negative pairs dynamically to optimize multi-modal latent space.
    """
    def __init__(self, scaling_a=2.0, scaling_b=50.0, threshold=0.5):
        super(MultiSimilarityLoss, self).__init__()
        self.scaling_a = scaling_a
        self.scaling_b = scaling_b
        self.threshold = threshold

    def forward(self, embeddings, labels):
        # Normalize embeddings for cosine similarity calculation
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        batch_size = embeddings.size(0)
        loss = 0.0
        valid_classes = 0

        for i in range(batch_size):
            label_i = labels[i]
            # Positive and negative masks
            pos_mask = (labels == label_i)
            pos_mask[i] = False # Exclude self
            neg_mask = (labels != label_i)

            if not pos_mask.any() or not neg_mask.any():
                continue

            valid_classes += 1
            s_p = sim_mat[i][pos_mask]
            s_n = sim_mat[i][neg_mask]

            # Hard mining & weighting terms
            alpha = self.scaling_a
            beta = self.scaling_b
            thresh = self.threshold

            pos_term = (1.0 / alpha) * torch.log(1.0 + torch.sum(torch.exp(-alpha * (s_p - thresh))))
            neg_term = (1.0 / beta) * torch.log(1.0 + torch.sum(torch.exp(beta * (s_n - thresh))))
            
            loss += pos_term + neg_term

        return loss / max(valid_classes, 1)

if __name__ == "__main__":
    ms_loss = MultiSimilarityLoss()
    dummy_emb = torch.randn(8, 512, requires_grad=True)
    dummy_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])
    loss_val = ms_loss(dummy_emb, dummy_labels)
    print(f"[Metric Loss Test] Multi-Similarity Loss calculated successfully: {loss_val.item():.4f}")