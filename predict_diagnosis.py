import torch
from torchvision import transforms
from PIL import Image
from fusion_model import CrossModalFusionModel
from multimodal_dataset import TOTAL_HPO_FEATURES, DISEASE_CLASSES, encode_hpo_vector
from knowledge_loader import OrphadataKnowledgeBase
import torch.nn.functional as F

def predict_patient_case(image_path):
    """
    Performs robust zero-shot multimodal inference using latent feature similarity matching.
    """
    # 1. Setup Device
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    # 2. Load Trained Model Weights
    model = CrossModalFusionModel(num_hpo_features=TOTAL_HPO_FEATURES, embedding_dim=512, num_classes=5)
    model.load_state_dict(torch.load("best_rare_disease_model.pth", map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # 3. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Latent Space Similarity Matching across all 5 class profiles
    with torch.no_grad():
        similarities = []
        
        # Extract visual representation directly
        for class_id in range(5):
            phenotype_tensor = encode_hpo_vector(class_id).unsqueeze(0).to(device)
            _, img_embed, text_embed, fused_embed = model(image_tensor, phenotype_tensor)
            
            # Compute cosine similarity between the visual embedding and the fused/text profile embedding
            sim = F.cosine_similarity(img_embed, fused_embed, dim=1).item()
            similarities.append((class_id, sim))

        # Normalize similarities into clean percentage confidences using softmax
        sim_tensor = torch.tensor([s[1] for s in similarities], device=device)
        probabilities = torch.softmax(sim_tensor * 5.0, dim=0) * 100  # Scaling factor to sharpen distribution

        all_results = []
        best_conf = -1.0
        best_class = None

        for idx, (class_id, _) in enumerate(similarities):
            conf = probabilities[idx].item()
            all_codes_id = similarities[idx][0]
            all_results.append((all_codes_id, conf))
            if conf > best_conf:
                best_conf = conf
                best_class = all_codes_id

    predicted_name = DISEASE_CLASSES[best_class]

    # Pull Orphadata genetic metadata for reporting
    kb = OrphadataKnowledgeBase()
    associated_genes = kb.get_disease_genes(predicted_name)

    print("\n" + "="*55)
    print("   MULTIMODAL RARE DISEASE DIAGNOSTIC REPORT   ")
    print("="*55)
    print(f"👉 Uploaded Scan: {image_path}")
    print(f"🎯 Optimal Predicted Condition: {predicted_name} (Class ID: {best_class})")
    print(f"📊 Model Confidence Score: {best_conf:.2f}%")
    print(f"🧬 Associated Causal Genes (Orphadata): {', '.join(associated_genes)}")
    print("-" * 55)
    print(" Candidate Breakdown:")
    for cid, score in all_results:
        print(f"   - {DISEASE_CLASSES[cid]}: {score:.2f}%")
    print("="*55)

if __name__ == "__main__":
    import os
    
    print("\n" + "="*55)
    print("   INTERACTIVE MULTIMODAL DIAGNOSTIC TESTER   ")
    print("="*55)
    
    user_input_path = input("Enter the file path of the MRI scan you want to test: ").strip()
    user_input_path = user_input_path.strip('"\'')
    
    if user_input_path and os.path.exists(user_input_path):
        predict_patient_case(user_input_path)
    else:
        print("\n[Error] The provided file path does not exist. Please check the path and try again.")