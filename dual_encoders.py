import torch
import torch.nn as nn
import torchvision.models as models

class DualEncoderModel(nn.Module):
    """
    Multimodal Dual-Encoder Framework:
    - Stream A (Vision Encoder): ResNet-50 backbone extracting 512-dim visual features z_img.
    - Stream B (Phenotype Encoder): Linear projection mapping multi-hot HPO vectors to z_graph.
    """
    def __init__(self, num_hpo_features, embedding_dim=512):
        super(DualEncoderModel, self).__init__()
        
        # -----------------------------------------------------------------
        # Stream A: Vision Encoder (ResNet-50 pre-trained on ImageNet)
        # -----------------------------------------------------------------
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Replace the final fully connected layer to output a 512-dimensional vector
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1]) # Outputs [Batch, 2048, 1, 1]
        self.vision_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # -----------------------------------------------------------------
        # Stream B: Phenotype / Knowledge Graph Encoder
        # -----------------------------------------------------------------
        self.phenotype_encoder = nn.Sequential(
            nn.Linear(num_hpo_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, images, phenotypes):
        # Stream A Forward
        x_vis = self.vision_backbone(images)
        z_img = self.vision_projection(x_vis)  # Shape: [Batch, 512]
        
        # Stream B Forward
        z_graph = self.phenotype_encoder(phenotypes)  # Shape: [Batch, 512]
        
        return z_img, z_graph

# -------------------------------------------------------------------------
# Verification Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from multimodal_dataset import get_dataloaders
    import os

    DATASET_PATH = r"D:\anu_research"
    
    # Setup DirectML Device
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[Hardware] Using AMD GPU via DirectML: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hardware] DirectML fallback: {device}")

    if os.path.exists(DATASET_PATH):
        train_loader, _ = get_dataloaders(DATASET_PATH, batch_size=4)
        
        # Get number of HPO features from the dataset's vocab size
        from multimodal_dataset import TOTAL_HPO_FEATURES
        
        model = DualEncoderModel(num_hpo_features=TOTAL_HPO_FEATURES, embedding_dim=512).to(device)
        
        # Test a single batch through the dual encoders
        for images, phenotypes, labels in train_loader:
            images = images.to(device)
            phenotypes = phenotypes.to(device)
            
            z_img, z_graph = model(images, phenotypes)
            
            print("\n[Dual Encoder Test Successful]")
            print(f"  - Vision Feature Vector (z_img):   {z_img.shape} (Expected: [4, 512])")
            print(f"  - Phenotype Vector (z_graph):      {z_graph.shape} (Expected: [4, 512])")
            break
    else:
        print("[Notice] Dataset path not found.")