import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BreaKHisMultiScaleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images 
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = [] 
        
        # BreaKHis classes
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        self.mags = ['40X', '100X', '200X', '400X']

        self._prepare_data()

    def _prepare_data(self):
        print(f"Scanning dataset directory: {self.root_dir.resolve()}...")
        
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} not found.")
                continue

            # Recursive search: Find every '40X' folder anywhere inside benign/malignant
            # This completely bypasses the issue of deeply nested tumor-type folders.
            for mag_40_dir in class_dir.rglob('40X'):
                # The parent of the 40X folder is the patient/slide folder
                patient_dir = mag_40_dir.parent
                
                patient_images = {mag: [] for mag in self.mags}
                valid_patient = True

                # Check if this patient folder has all 4 magnification subfolders
                for mag in self.mags:
                    mag_dir = patient_dir / mag
                    if mag_dir.exists():
                        imgs = list(mag_dir.glob('*.png')) + list(mag_dir.glob('*.jpg'))
                        if len(imgs) > 0:
                            patient_images[mag] = imgs
                        else:
                            valid_patient = False # Missing images
                            break
                    else:
                        valid_patient = False # Missing magnification folder
                        break

                # If the patient has valid data in all 4 scales, build the samples
                if valid_patient:
                    # We use 40X as our "anchor"
                    for img_40x_path in patient_images['40X']:
                        self.samples.append({
                            'base_40x': img_40x_path,
                            'patient_images': patient_images,
                            'label': label
                        })
                        
        print(f"Successfully loaded {len(self.samples)} multi-scale sample groups.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        patient_images = sample['patient_images']

        img_40x_path = sample['base_40x']

        # Dynamically sample ONE random image from the other scales for this SAME patient.
        img_100x_path = random.choice(patient_images['100X'])
        img_200x_path = random.choice(patient_images['200X'])
        img_400x_path = random.choice(patient_images['400X'])

        # Load images
        img_40 = Image.open(img_40x_path).convert('RGB')
        img_100 = Image.open(img_100x_path).convert('RGB')
        img_200 = Image.open(img_200x_path).convert('RGB')
        img_400 = Image.open(img_400x_path).convert('RGB')

        # Apply transforms
        if self.transform:
            img_40 = self.transform(img_40)
            img_100 = self.transform(img_100)
            img_200 = self.transform(img_200)
            img_400 = self.transform(img_400)

        images_dict = {
            '40X': img_40, 
            '100X': img_100, 
            '200X': img_200, 
            '400X': img_400
        }
        
        return images_dict, label

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Corrected path based on your exact command output
    DATASET_PATH = "../dataset/BreaKHis_v1"
    
    try:
        dataset = BreaKHisMultiScaleDataset(root_dir=DATASET_PATH, transform=transform)
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            for batch_images, batch_labels in dataloader:
                print("\nBatch loaded successfully!")
                print(f"40X Tensor Shape: {batch_images['40X'].shape}")
                print(f"Labels: {batch_labels}")
                break 
        else:
            print("\nError: Still found 0 samples. Please double-check that the data_raw folder contains the actual images.")
            
    except Exception as e:
        print(f"An error occurred: {e}")