import torch
import torch.nn as nn
from dual_encoders import DualEncoderModel

class CrossModalFusionModel(nn.Module):
    """
    Multimodal Framework with Cross-Attention Fusion:
    Takes the visual vector (z_img) and phenotype vector (z_graph) 
    and fuses them using multi-head cross-attention.
    """
    def __init__(self, num_hpo_features, embedding_dim=512, num_classes=5):
        super(CrossModalFusionModel, self).__init__()
        
        # 1. Load Dual Encoders
        self.encoder = DualEncoderModel(num_hpo_features, embedding_dim)
        
        # 2. Cross-Attention Layer (treating image as query, phenotype as key/value or vice versa)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        
        # 3. Final Classification / Metric Head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, phenotypes):
        # Step A & B: Extract 512-dim features
        z_img, z_graph = self.encoder(images, phenotypes) # Each shape: [Batch, 512]
        
        # Reshape for multi-head attention expecting [Batch, Seq_Len=1, Dim]
        q = z_img.unsqueeze(1)    # Query: Vision
        k = z_graph.unsqueeze(1)  # Key: Phenotype
        v = z_graph.unsqueeze(1)  # Value: Phenotype
        
        # Apply Cross-Attention (Visual features attending to Clinical Phenotypes)
        attn_output, _ = self.cross_attention(q, k, v)
        z_fused_attention = attn_output.squeeze(1) # Shape: [Batch, 512]
        
        # Concatenate original representation with attention output for robust feature fusion
        combined = torch.cat([z_img, z_fused_attention], dim=1) # Shape: [Batch, 1024]
        
        # Classification logits
        logits = self.classifier(combined)
        
        return logits, z_img, z_graph, z_fused_attention

# -------------------------------------------------------------------------
# Verification Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from multimodal_dataset import get_dataloaders, TOTAL_HPO_FEATURES
    import os

    DATASET_PATH = r"D:\anu_research"
    
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[Hardware] Using AMD GPU via DirectML: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hardware] DirectML fallback: {device}")

    if os.path.exists(DATASET_PATH):
        train_loader, _ = get_dataloaders(DATASET_PATH, batch_size=4)
        
        model = CrossModalFusionModel(num_hpo_features=TOTAL_HPO_FEATURES, embedding_dim=512, num_classes=5).to(device)
        
        for images, phenotypes, labels in train_loader:
            images = images.to(device)
            phenotypes = phenotypes.to(device)
            labels = labels.to(device)
            
            logits, z_img, z_graph, z_fused = model(images, phenotypes)
            
            print("\n[Cross-Modal Fusion Test Successful]")
            print(f"  - Logits Output Shape: {logits.shape} (Expected: [4, 5])")
            print(f"  - Fused Feature Shape: {z_fused.shape} (Expected: [4, 512])")
            break