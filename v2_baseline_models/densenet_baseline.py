import os
import random
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- 1. SINGLE-SCALE DATASET (Targeting 200X as requested) ---
class BreaKHisSingleScaleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        self._prepare_data()

    def _prepare_data(self):
        print("Scanning for 200X anchor images to match ResNet baseline...")
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists(): continue
            
            # Explicitly searching for 200X to match your previous run
            for mag_dir in class_dir.rglob('200X'):
                imgs = list(mag_dir.glob('*.png')) + list(mag_dir.glob('*.jpg'))
                for img_path in imgs:
                    self.samples.append({'path': img_path, 'label': label})

        # CRITICAL: Sort so seed(42) produces the EXACT same split
        self.samples.sort(key=lambda x: str(x['path']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, sample['label']

# --- 2. DENSENET121 ARCHITECTURE ---
class BaselineDenseNet(nn.Module):
    def __init__(self):
        super(BaselineDenseNet, self).__init__()
        
        # Load pre-trained DenseNet121 (The medical imaging gold standard)
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Freeze the early feature blocks to save memory and retain basic image logic
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # Unfreeze the very last DenseBlock (block4) and norm5 for fine-tuning
        for param in self.model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.model.features.norm5.parameters():
            param.requires_grad = True
            
        # Replace the final classifier (DenseNet121 defaults to 1024 -> 1000)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Output raw logit
        )

    def forward(self, x):
        return self.model(x)

# --- 3. TRAINING & EVALUATION PIPELINE ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create Results Directory for DenseNet
    results_dir = "results_densenet"
    os.makedirs(results_dir, exist_ok=True) 

    # Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 70/15/15 Data Split
    dataset_path = "../dataset/BreaKHis_v1"
    full_dataset = BreaKHisSingleScaleDataset(dataset_path)
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    train_idx, val_idx, test_idx = indices[:1400], indices[1400:1700], indices[1700:]
    
    # Batch size 32
    train_loader = DataLoader(Subset(BreaKHisSingleScaleDataset(dataset_path, train_transform), train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(BreaKHisSingleScaleDataset(dataset_path, val_test_transform), val_idx), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(BreaKHisSingleScaleDataset(dataset_path, val_test_transform), test_idx), batch_size=32, shuffle=False)

    # Init Model
    model = BaselineDenseNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    
    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # EARLY STOPPING SETTINGS
    epochs = 30
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    model_save_path = os.path.join(results_dir, "best_densenet.pth")

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item()
                preds = torch.sigmoid(outputs) > 0.5
                v_corr += (preds == labels).sum().item()
                v_tot += labels.size(0)
                
        val_losses.append(v_loss / len(val_loader))
        val_accs.append(v_corr / v_tot)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]*100:.2f}%")
        
        # EARLY STOPPING LOGIC
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Model improved! Saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
                break

    print("\n--- Generating Plots & Final Evaluation ---")
    
    # 1. Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    plt.savefig(os.path.join(results_dir, "learning_curves.png"))

    # 2. Test Set Evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title("DenseNet121 - Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right"); plt.title("DenseNet121 - ROC Curve")
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))

    print("\n=== DenseNet121 Baseline Complete! ===")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

if __name__ == "__main__":
    main()