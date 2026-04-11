import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import random
import os
import matplotlib.pyplot as plt

from data_loader import BreaKHisMultiScaleDataset
from model import MultiScaleBreastCancerModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Create Results Directory ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    BATCH_SIZE = 16 
    EPOCHS = 30
    LEARNING_RATE = 1e-4

    # --- THE ULTIMATE MERGE: Data Augmentation ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_path = "../dataset/BreaKHis_v1"
    full_train_dataset = BreaKHisMultiScaleDataset(root_dir=dataset_path, transform=train_transform)
    full_val_dataset = BreaKHisMultiScaleDataset(root_dir=dataset_path, transform=val_transform)

    # Safe Train/Val Split ensuring Train gets Augmented and Val stays Clean
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    # Train gets the first 1400
    train_indices = indices[:1400]
    
    # Val gets the next 300 (indices 1400 to 1700)
    val_indices = indices[1400:1700]
    
    # NOTE: We deliberately IGNORE the last 295 images. 
    # They are locked away for the evaluate.py script.

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    # Initialize Model
    model = MultiScaleBreastCancerModel(freeze_backbone=True).to(device)

    # Calculate Weights for Imbalanced Classes (Using instant CPU list reading)
    all_labels = [sample['label'] for sample in full_train_dataset.samples]
    num_pos = all_labels.count(1)
    num_neg = all_labels.count(0)
    
    print(f"Positive samples: {num_pos} | Negative samples: {num_neg}")
    
    # --- THE BUG FIX: Weighted Loss ---
    # Convert ratio to a tensor and actually pass it into the criterion!
    pos_weight_val = num_neg / num_pos
    pos_weights = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights) 
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # --- THE ULTIMATE MERGE: Early Stopping ---
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience = 5   # Bumped up to 5 because Data Augmentation causes loss to fluctuate more
    counter = 0    

    # --- Track History for Plots ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images_dict, labels) in enumerate(train_loader):
            images_dict = {scale: img.to(device) for scale, img in images_dict.items()}
            labels = labels.float().unsqueeze(1).to(device) # Required shape for BCE

            optimizer.zero_grad()

            outputs, attn_weights = model(images_dict)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images_dict, labels in val_loader:
                images_dict = {scale: img.to(device) for scale, img in images_dict.items()}
                labels = labels.float().unsqueeze(1).to(device)

                outputs, _ = model(images_dict)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Append to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        # Save BEST model (accuracy)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), "best_multiscale_model.pth")
            print(">>> New best model saved (accuracy)! <<<")

        # Early Stopping (validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  
            torch.save(model.state_dict(), "best_model_loss.pth")
            print(">>> Validation loss improved, model saved! <<<")
        else:
            counter += 1
            print(f"No improvement in val loss for {counter} epoch(s)")

        if counter >= patience:
            print("⛔ Early stopping triggered! The model has stopped learning.")
            break

    # --- PLOT LEARNING CURVES ---
    print("\nSaving learning curves to results/ directory...")
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "learning_curves.png"))

if __name__ == "__main__":
    train_model()