import os
import shutil
from pathlib import Path

def reorganize_breakhis(raw_dir, output_dir):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    print(f"Reading raw data from: {raw_path.resolve()}")
    print(f"Creating clean structure at: {out_path.resolve()}")

    # Ensure output directories exist
    for cls in ['benign', 'malignant']:
        (out_path / cls).mkdir(parents=True, exist_ok=True)

    # Count for progress
    patients_processed = 0

    # Search for all 40X folders to locate patients
    for mag_dir in raw_path.rglob('40X'):
        # The parent of the 40X folder is the patient ID (e.g., SOB_B_A_14-22549AB)
        patient_id = mag_dir.parent.name
        
        # Determine if this patient is benign or malignant by checking the path
        class_name = 'benign' if 'benign' in mag_dir.parts else 'malignant'
        
        # Create the new clean path: dataset/BreaKHis_v1/benign/SOB_.../
        patient_out_dir = out_path / class_name / patient_id
        
        # Copy all 4 magnifications over
        for mag in ['40X', '100X', '200X', '400X']:
            src_mag_dir = mag_dir.parent / mag
            dest_mag_dir = patient_out_dir / mag
            
            if src_mag_dir.exists():
                dest_mag_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                for img_file in src_mag_dir.glob('*'):
                    if img_file.is_file():
                        # shutil.copy2 preserves metadata. 
                        # Change to shutil.move if you want to DELETE the raw files as you go to save space.
                        shutil.copy2(img_file, dest_mag_dir / img_file.name)
        
        patients_processed += 1
        if patients_processed % 10 == 0:
            print(f"Processed {patients_processed} patients...")

    print(f"\nDone! Successfully reorganized {patients_processed} patients into {out_path.resolve()}")

if __name__ == "__main__":
    # Point these to your actual folders
    RAW_DATA_DIR = "./data_raw/BreaKHis_v1/histology_slides/breast"
    CLEAN_OUTPUT_DIR = "./dataset/BreaKHis_v1"
    
    reorganize_breakhis(RAW_DATA_DIR, CLEAN_OUTPUT_DIR)