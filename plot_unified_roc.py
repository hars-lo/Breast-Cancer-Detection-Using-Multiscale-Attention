import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# --- 1. DIRECT MODULE IMPORTS (BASELINES ONLY) ---
from v2_baseline_models.custom_cnn import UserCustomCNN
from v2_baseline_models.resnet50_baseline import BaselineResNet50
from v2_baseline_models.densenet_baseline import BaselineDenseNet

# --- 2. SINGLE SCALE DATASET (For Baselines Only) ---
class BreaKHisSingleScaleDataset(Dataset):
    def __init__(self, root_dir, target_mag, transform=None):
        self.root_dir = Path(root_dir)
        self.target_mag = target_mag
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        self._prepare_data()

    def _prepare_data(self):
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists(): continue
            for mag_dir in class_dir.rglob(self.target_mag):
                imgs = list(mag_dir.glob('*.png')) + list(mag_dir.glob('*.jpg'))
                for img_path in imgs:
                    self.samples.append({'path': img_path, 'label': label})
        self.samples.sort(key=lambda x: str(x['path']))

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.samples[idx]['label']

def get_test_loader(dataset):
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    test_idx = indices[1700:] # The exact baseline split
    return DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)

# --- 3. MAIN EVALUATION ENGINE ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pure evaluation on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_path = "dataset/BreaKHis_v1" 
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load the pristine V1 results (No Data Leakage!)
    print("Loading original Multi-Scale predictions...")
    v1_path = os.path.join(base_dir, "v1_multiscale_attention",  "v1_results.pt")
    
    if not os.path.exists(v1_path):
        print(f"CRITICAL ERROR: Cannot find {v1_path}. Please run Step 1 first!")
        return
        
    v1_data = torch.load(v1_path, weights_only=False)
    results = {
        "Multi-Scale Attention": {"true": v1_data['y_true'], "probs": v1_data['y_prob']}
    }

    # 2. Setup isolated Baseline loaders
    print("Building strict baseline splits...")
    loader_40x = get_test_loader(BreaKHisSingleScaleDataset(dataset_path, '40X', transform))
    loader_200x = get_test_loader(BreaKHisSingleScaleDataset(dataset_path, '200X', transform))

    models_config = {
        "DenseNet121 (200X)": {
            "model": BaselineDenseNet(),
            "path": os.path.join(base_dir, "v2_baseline_models", "results_densenet", "best_densenet.pth"),
            "loader": loader_200x
        },
        "ResNet50 (200X)": {
            "model": BaselineResNet50(),
            "path": os.path.join(base_dir, "v2_baseline_models", "results_resnet50", "best_resnet50.pth"),
            "loader": loader_200x
        },
        "Custom CNN (40X)": {
            "model": UserCustomCNN(),
            "path": os.path.join(base_dir, "v2_baseline_models", "results_custom_cnn", "best_custom_cnn.pth"),
            "loader": loader_40x
        }
    }

    # 3. Run isolated baseline inference
    for name, config in models_config.items():
        print(f"Evaluating {name}...")
        model = config["model"]
        model.load_state_dict(torch.load(config["path"], map_location=device))
        model.to(device).eval()
        
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch_imgs, labels in config["loader"]:
                y_true.extend(labels.numpy())
                gpu_imgs = batch_imgs.to(device)
                out = model(gpu_imgs)
                y_probs.extend(torch.sigmoid(out).cpu().numpy())
                
        results[name] = {"true": y_true, "probs": y_probs}

    # --- 4. PLOTTING THE MASTER ROC CURVE ---
    print("\nGenerating clean, leak-free ROC plot...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    colors = {"Multi-Scale Attention": "#E63946", "DenseNet121 (200X)": "#1D3557", "ResNet50 (200X)": "#457B9D", "Custom CNN (40X)": "#F4A261"}
    line_widths = {"Multi-Scale Attention": 3, "DenseNet121 (200X)": 2, "ResNet50 (200X)": 2, "Custom CNN (40X)": 2}

    for name in results.keys():
        fpr, tpr, _ = roc_curve(results[name]["true"], results[name]["probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[name], lw=line_widths[name], label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Comparative ROC Analysis (Strict Test Isolation)', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    output_path = "comparative_roc_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Master ROC curve saved to: {output_path}")

if __name__ == "__main__":
    main()