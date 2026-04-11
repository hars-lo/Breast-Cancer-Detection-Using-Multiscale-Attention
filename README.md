# 🔬 Multi-Scale Breast Cancer Diagnostic AI
This repository contains a Deep Learning pipeline and interactive web dashboard for classifying breast cancer histopathology images. It utilizes a custom **Multi-Scale Attention Architecture** built on top of ResNet50 to analyze tissue slides across four different magnifications (40X, 100X, 200X, 400X) simultaneously.

### 🏆 Model Performance
Tested on a strictly isolated, unseen test split of the **BreaKHis v1 Dataset**:
* **Accuracy:** 99.00%
* **ROC-AUC:** 0.998
* **Malignant Recall:** 99.00% (Critical for minimizing false negatives in medical diagnostics)

---

## 📁 Project Structure

```text
├── v1_multiscale_attention/
│   ├── app.py                           # Streamlit Web Dashboard
│   ├── src/                             # Core Deep Learning logic
│   │   ├── data_loader.py               # Handles recursive multi-scale image loading
│   │   ├── model.py                     # Custom Multi-Scale Attention Neural Network
│   │   ├── train.py                     # Training loop with Early Stopping & Augmentation
│   │   └── evaluate.py                  # Generates Confusion Matrix & ROC curves
│   ├── models/                          # Directory for .pth weight files
│   └── results/                         # Generated performance graphs
├── reorganise_dataset.py                # Script to clean raw BreaKHis data
├── requirements.txt                     # Python dependencies
└── README.md

SETUP:-
Step 1: Clone the Repository
Open your terminal or command prompt and run:

Bash
git clone [insert repo link]
cd breakhis-multiscale-attention


Step 2: Create a Virtual Environment
For Windows:
Bash
python -m venv venv
.\venv\Scripts\activate
For Mac/Linux:
Bash
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
With your virtual environment activated, install the required packages:

Bash
pip install -r requirements.txt
(Note: If you have an NVIDIA GPU, you may want to install the CUDA-specific version of PyTorch from their official website for much faster inference).

Step 4: Download the Pre-Trained Weights
Because Deep Learning weight files are too large for standard GitHub tracking, they are hosted in the Releases tab.

Go to the Releases tab on this GitHub page.

Download the file named best_multiscale_model.pth.

Move that file into the v1_multiscale_attention/models/ folder on your computer.
