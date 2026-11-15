import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from knowledge_loader import OrphadataKnowledgeBase

# -------------------------------------------------------------------------
# 1. STANDARDIZED CLINICAL PHENOTYPE MAPPINGS (From Disease Details)
# -------------------------------------------------------------------------
# Define the standardized HPO terms and diagnostic signatures for each target disease
DISEASE_CLASSES = {
    0: "Walker-Warburg Syndrome",
    1: "Pachygyria",
    2: "Moyamoya Disease",
    3: "Hallervorden-Spatz Disease",  # Also known as PKAN
    4: "Fukuyama Muscular Dystrophy"
}

# Directory name mapping in your preprocessed Mendeley dataset folder
FOLDER_NAME_MAP = {
    "Walker-Warburg Syndrome": 0,
    "Walker Warburg": 0,
    "Pachygyria": 1,
    "Pachygyria with Cerebellar Hypoplasia": 1,
    "Moyamoya Disease": 2,
    "Moyamoya": 2,
    "Hallervorden-Spatz Disease": 3,
    "Hallervorden Spatz": 3,
    "PKAN": 3,
    "Fukuyama Muscular Dystrophy": 4,
    "Fukuyama Congenital Muscular Dystrophy": 4
}

# Initialize knowledge base for dynamic gene fetching
kb = OrphadataKnowledgeBase()

# Enhanced HPO + Gene Knowledge Signatures for your 5 Classes
DISEASE_KNOWLEDGE_SIGNATURES = {
    0: {"name": "Walker-Warburg Syndrome", "hpo": ["HP:0001339", "HP:0001321", "HP:0002060"], "genes": kb.get_disease_genes("Walker-Warburg")},
    1: {"name": "Pachygyria", "hpo": ["HP:0001302", "HP:0001321", "HP:0002197"], "genes": kb.get_disease_genes("Pachygyria")},
    2: {"name": "Moyamoya Disease", "hpo": ["HP:0002633", "HP:0001297", "HP:0002170"], "genes": kb.get_disease_genes("Moyamoya")},
    3: {"name": "Hallervorden-Spatz Disease (PKAN)", "hpo": ["HP:0010884", "HP:0001332", "HP:0002061"], "genes": kb.get_disease_genes("Hallervorden")},
    4: {"name": "Fukuyama Muscular Dystrophy", "hpo": ["HP:0001339", "HP:0002197", "HP:0003701"], "genes": kb.get_disease_genes("Fukuyama")}
}

# Standardized HPO Symptoms & Key Imaging Signatures
HPO_PHENOTYPE_DICT = {
    0: [  # Walker-Warburg Syndrome (POMT1/POMT2/FKTN)
        "HP:0001339",  # Cobblestone lissencephaly
        "HP:0001321",  # Cerebellar hypoplasia
        "HP:0002060",  # Hydrocephalus
        "HP:0001252",  # Muscular hypotonia
        "HP:0000518",  # Cataracts / Retinal dysplasia
        "HP:0001263",  # Global developmental delay
    ],
    1: [  # Pachygyria with Cerebellar Hypoplasia (LIS1/DCX)
        "HP:0001302",  # Pachygyria (broad, flat gyri)
        "HP:0001321",  # Cerebellar hypoplasia
        "HP:0002197",  # Generalized seizures / Epilepsy
        "HP:0001252",  # Hypotonia
        "HP:0001263",  # Severe intellectual disability
        "HP:0001249",  # Motor developmental delay
    ],
    2: [  # Moyamoya Disease with Hemorrhage (RNF213)
        "HP:0002633",  # Moyamoya collateral vessel network ("puff of smoke")
        "HP:0001297",  # Transient ischemic attacks (TIA)
        "HP:0002170",  # Intraventricular hemorrhage
        "HP:0002315",  # Recurrent stroke
        "HP:0001259",  # Coma / Altered consciousness
        "HP:0002321",  # Vertigo
    ],
    3: [  # Hallervorden-Spatz Disease / PKAN (PANK2)
        "HP:0010884",  # "Eye-of-the-tiger" sign in globus pallidus
        "HP:0001332",  # Dystonia
        "HP:0002061",  # Rigidity and spasticity
        "HP:0002072",  # Choreoathetosis
        "HP:0000543",  # Retinal degeneration / Pigmentary retinopathy
        "HP:0002167",  # Progressive neurological deterioration
    ],
    4: [  # Fukuyama Muscular Dystrophy (FKTN)
        "HP:0001339",  # Cobblestone lissencephaly (type II)
        "HP:0002197",  # Febrile and non-febrile seizures
        "HP:0003701",  # High serum creatine kinase (>10x)
        "HP:0003198",  # Generalized muscle weakness / Dystrophy
        "HP:0001250",  # Pseudohypertrophy of calves
        "HP:0001263",  # Severe psychomotor retardation
    ]
}

# -------------------------------------------------------------------------
# 2. VOCABULARY BUILDER FOR HPO MULTI-HOT ENCODING
# -------------------------------------------------------------------------
def build_hpo_vocab():
    """Extracts unique HPO codes to build a one-hot / multi-hot feature vector."""
    unique_hpos = sorted(list({hpo for hpo_list in HPO_PHENOTYPE_DICT.values() for hpo in hpo_list}))
    hpo_to_idx = {hpo: idx for idx, hpo in enumerate(unique_hpos)}
    return hpo_to_idx, len(unique_hpos)

HPO_TO_IDX, TOTAL_HPO_FEATURES = build_hpo_vocab()

def encode_hpo_vector(disease_id):
    """Converts a disease's HPO terms into a binary multi-hot feature tensor."""
    vector = torch.zeros(TOTAL_HPO_FEATURES, dtype=torch.float32)
    for hpo in HPO_PHENOTYPE_DICT[disease_id]:
        if hpo in HPO_TO_IDX:
            vector[HPO_TO_IDX[hpo]] = 1.0
    return vector

# -------------------------------------------------------------------------
# 3. PYTORCH MULTIMODAL DATASET
# -------------------------------------------------------------------------
class RareDiseaseMultimodalDataset(Dataset):
    """
    Multimodal Dataset class that pairs:
    1. Stream A: Preprocessed MRI Scans (Tensor: [C, H, W])
    2. Stream B: Phenotype Feature Vector (Tensor: [Num_HPO_Features])
    3. Label: Disease Class Index (0-4)
    """
    def __init__(self, data_root_dir, transform=None):
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.samples = []  # List of tuples: (image_path, class_idx)

        # Scan directory and match class folders
        for folder_name in os.listdir(data_root_dir):
            folder_path = os.path.join(data_root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Skip system and metadata folders silently (using substring check)
            if 'metadata' in folder_name.lower() or '__pycache__' in folder_name.lower():
                continue

            # Identify class ID from folder name
            matched_class_id = None
            for key, val in FOLDER_NAME_MAP.items():
                if key.lower() in folder_name.lower():
                    matched_class_id = val
                    break

            if matched_class_id is not None:
                # Support standard medical image extensions
                image_paths = []
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'):
                    image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

                for img_p in image_paths:
                    self.samples.append((img_p, matched_class_id))
            else:
                print(f"[Warning] Folder '{folder_name}' did not match any of the 5 rare disease targets. Skipping.")

        # print(f"\n[Dataset Initialized] Total Paired Samples: {len(self.samples)}")
        # for class_id, class_name in DISEASE_CLASSES.items():
        #     count = sum(1 for _, cid in self.samples if cid == class_id)
        #     print(f"  - Class {class_id} ({class_name}): {count} scans")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]

        # 1. Load MRI image (ensure 3 channels for pre-trained vision backbones)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        # 2. Get structured clinical phenotype vector (Stream B)
        phenotype_tensor = encode_hpo_vector(class_id)

        # 3. Label
        label = torch.tensor(class_id, dtype=torch.long)

        return image_tensor, phenotype_tensor, label

# -------------------------------------------------------------------------
# 4. DATA LOADER BUILDER & HARDWARE VERIFICATION
# -------------------------------------------------------------------------
def get_dataloaders(data_dir, batch_size=16, train_split=0.8):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset once to get indices
    temp_dataset = RareDiseaseMultimodalDataset(data_root_dir=data_dir, transform=None)
    train_size = int(train_split * len(temp_dataset))
    val_size = len(temp_dataset) - train_size
    
    indices = torch.randperm(len(temp_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create separate subset datasets with their respective transforms
    from torch.utils.data import Subset
    
    train_raw = RareDiseaseMultimodalDataset(data_root_dir=data_dir, transform=train_transform)
    val_raw = RareDiseaseMultimodalDataset(data_root_dir=data_dir, transform=val_transform)

    train_dataset = Subset(train_raw, train_indices)
    val_dataset = Subset(val_raw, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

if __name__ == "__main__":
    # Specify the path where your preprocessed disease folders are stored
    DATASET_PATH = r"D:\anu_research"  # <--- Update if your folders are located elsewhere

    # Check DirectML (AMD GPU) Device
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[Hardware] Using AMD GPU via DirectML: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hardware] DirectML not found. Falling back to: {device}")

    if os.path.exists(DATASET_PATH):
        train_loader, val_loader = get_dataloaders(DATASET_PATH, batch_size=8)

        # Test a single batch
        for images, phenotypes, labels in train_loader:
            images = images.to(device)
            phenotypes = phenotypes.to(device)
            labels = labels.to(device)

            print("\n[Batch Test Successful]")
            print(f"  - MRI Image Batch Shape:      {images.shape}")
            print(f"  - Phenotype Vector Batch Shape: {phenotypes.shape}")
            print(f"  - Labels:                     {labels.tolist()}")
            break
    else:
        print(f"[Notice] Directory '{DATASET_PATH}' does not exist yet. Please check your path.")