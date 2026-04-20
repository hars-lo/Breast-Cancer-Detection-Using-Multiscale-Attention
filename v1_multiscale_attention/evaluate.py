import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from data_loader import BreaKHisMultiScaleDataset
from model import MultiScaleBreastCancerModel

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Create Results Directory ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = BreaKHisMultiScaleDataset(root_dir="../dataset/BreaKHis_v1", transform=transform)
    
    # --- Proper Random Shuffle ---
    # Create a list of all indices, shuffle them, and grab the last 300
    indices = list(range(len(dataset)))
    random.seed(42) # Keeps the random shuffle identical every time you run this script
    random.shuffle(indices)
    # Grab the exact 295 images that train.py completely ignored
    test_indices = indices[1700:]
    
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 2. Load the trained model
    # Ensure num_classes=1 matches our new BCE loss setup
    model = MultiScaleBreastCancerModel(num_classes=1, freeze_backbone=True).to(device)
    try:
        model.load_state_dict(torch.load("best_multiscale_model.pth"))
        print("Loaded best_multiscale_model.pth successfully.")
    except FileNotFoundError:
        print("ERROR: Model weights not found. You need to run train.py first!")
        return

    model.eval()

    all_preds = []
    all_labels = []
    all_attention = []
    all_probs = [] # Added for ROC curve

    print("Evaluating properly mixed test set...")
    with torch.no_grad():
        for images_dict, labels in test_loader:
            images_dict = {scale: img.to(device) for scale, img in images_dict.items()}
            
            outputs, attn_weights = model(images_dict)
            
            # --- THE FIX FOR BCE LOSS ---
            # Convert raw logit into a probability (0.0 to 1.0)
            probs = torch.sigmoid(outputs)
            # Threshold at 0.5 to get 0 (Benign) or 1 (Malignant), and flatten the array
            preds = (probs > 0.5).int().view(-1) 
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            all_attention.append(attn_weights.cpu().numpy())

    # 3. Print Metrics
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

    # 4. Visualize Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    print(f"Saved {results_dir}/confusion_matrix.png")

    # 5. Visualize "Novel" Attention Weights
    avg_attention = np.mean(np.concatenate(all_attention, axis=0), axis=0).flatten()
    scales = ['40X', '100X', '200X', '400X']

    plt.figure(figsize=(8, 5))
    sns.barplot(x=scales, y=avg_attention, hue=scales, legend=False, palette='viridis')
    plt.title("Which Magnification Did the Model Rely On?")
    plt.ylabel("Average Attention Weight (Sums to 1.0)")
    plt.xlabel("Magnification Scale")
    plt.ylim(0, 1)
    for i, v in enumerate(avg_attention):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "attention_weights.png"))
    print(f"Saved {results_dir}/attention_weights.png")

    # 6. Visualize ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    print(f"Saved {results_dir}/roc_curve.png")

    # Add this at the end of your evaluate.py script
    torch.save({'y_true': all_labels, 'y_prob': all_probs}, 'v1_results.pt')
    print("Saved true V1 predictions to v1_results.pt!")



if __name__ == "__main__":
    evaluate_model()