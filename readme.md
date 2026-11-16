# Multimodal Knowledge-Guided Rare Disease Diagnostic Framework

An advanced deep learning framework designed for B.Tech Honors research to classify rare neurological and genetic brain disorders using a dual-stream architecture combining **MRI neuroimaging** and **Human Phenotype Ontology (HPO) / Orphadata clinical knowledge graphs**.

---

## 🧠 System Architecture & Methodology

The framework bridges visual imaging data with structured biomedical knowledge through a multi-stream fusion model:
1. **Stream A (Visual Backbone):** A fine-tuned **ResNet-50** deep convolutional neural network that extracts spatial feature maps and morphological abnormalities from brain MRI scans (e.g., agyria, ventricular enlargement, structural white matter changes).
2. **Stream B (Knowledge Graph Stream):** Encodes disease-specific clinical phenotype vectors derived from HPO and Orphadata genetic metadata.
3. **Cross-Attention Fusion:** Dynamically projects visual and textual representations into a unified 512-dimensional latent space using cross-modal attention mechanisms to align imaging findings with established genetic/phenotypic profiles.

---

## 📂 Repository File Structure

* **`fusion_model.py`**: Defines the `CrossModalFusionModel` architecture combining ResNet-50 visual features with HPO phenotype embeddings via cross-attention layers.
* **`multimodal_dataset.py`**: Manages dataset loading, preprocessing pipelines, and HPO feature vector encoding (`encode_hpo_vector`).
* **`knowledge_loader.py`**: Interfaces with Orphadata metadata files to retrieve associated causal genes (e.g., *PANK2*, *POMT1*, *POMT2*, *FKTN*) for each predicted condition.
* **`predict_diagnosis.py`**: Interactive inference script allowing users to input an MRI scan and generate a comprehensive diagnostic report with confidence scores and genetic metadata.

---

## 🚀 Supported Disease Categories

1. **Walker-Warburg Syndrome** *(Class ID: 0 | Causal Genes: POMT1, POMT2, FKTN)*
2. **Pachygyria** *(Class ID: 1)*
3. **Moyamoya Disease** *(Class ID: 2)*
4. **Hallervorden-Spatz Disease / PKAN** *(Class ID: 3 | Causal Genes: PANK2)*
5. **Fukuyama Muscular Dystrophy** *(Class ID: 4)*

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/anu-singh2507/Research_Work.git
cd Research_Work
```

### 2. Set Up a Conda Environment
```powershell
conda create -n rare_disease_ai python=3.10
conda activate rare_disease_ai
```

### 3. Install Dependencies
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy pandas scikit-learn
```
*(Note: If running on Windows with AMD hardware, you can optionally install torch-directml for local GPU acceleration).*

---

## 🧪 Running Diagnostic Inference
To run an interactive test on an unseen brain MRI scan:
```powershell
python predict_diagnosis.py
```
*When prompted, paste the file path of your test MRI image. The script will preprocess the image, evaluate it through the multimodal network, and output a structured diagnostic report detailing the predicted condition, confidence score, and associated Orphadata causal genes.*

---

## 📜 Academic Context
**Developed as part of undergraduate biomedical machine learning research focusing on multimodal rare disease diagnostic models, few-shot learning methods, and knowledge-guided neural network integration.**
