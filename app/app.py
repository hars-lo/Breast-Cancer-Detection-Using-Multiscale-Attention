import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# --- PATH WIRING ---
# This tells Python to look inside the v1_multiscale_attention folder for your code
current_dir = os.path.dirname(os.path.abspath(__file__))
v1_dir = os.path.join(current_dir, "..", "v1_multiscale_attention")
sys.path.append(v1_dir)

# THE FIX 1: Import directly from model.py since it is not in a src/ folder!
from model import MultiScaleBreastCancerModel

# --- PAGE SETUP ---
st.set_page_config(page_title="BreaKHis Diagnostic AI", page_icon="🔬", layout="wide")

st.title("🔬 Multi-Scale Breast Cancer Diagnostic AI")
st.markdown("""
Upload the 4 corresponding magnification scans (40X, 100X, 200X, 400X) of the patient's tissue sample. 
The Multi-Scale Attention network will fuse the features to provide a clinical diagnosis.
""")

# --- LOAD MODEL (Cached so it only loads into memory once) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleBreastCancerModel(num_classes=1, freeze_backbone=True)
    
    # THE FIX 2: Look directly in v1_dir instead of looking for a models/ folder!
    model_path = os.path.join(v1_dir, "best_multiscale_model.pth")
    
    if not os.path.exists(model_path):
        st.error(f"Cannot find model weights at: {model_path}")
        return None, device
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# --- IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- UI: THE 4 UPLOADERS ---
st.markdown("### Patient Scans")
col1, col2, col3, col4 = st.columns(4)

with col1:
    img_40 = st.file_uploader("Upload 40X Scan", type=["png", "jpg", "jpeg"])
with col2:
    img_100 = st.file_uploader("Upload 100X Scan", type=["png", "jpg", "jpeg"])
with col3:
    img_200 = st.file_uploader("Upload 200X Scan", type=["png", "jpg", "jpeg"])
with col4:
    img_400 = st.file_uploader("Upload 400X Scan", type=["png", "jpg", "jpeg"])

# --- INFERENCE ENGINE ---
# Only run if the user has uploaded all 4 images
if img_40 and img_100 and img_200 and img_400:
    st.success("All 4 magnifications loaded. Ready for analysis.")
    
    if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
        if model is None:
            st.error("Model failed to load. Please check your file paths.")
        else:
            with st.spinner("Analyzing cellular structures across multiple scales..."):
                
                # 1. Open and convert images to RGB
                images = {
                    '40X': Image.open(img_40).convert('RGB'),
                    '100X': Image.open(img_100).convert('RGB'),
                    '200X': Image.open(img_200).convert('RGB'),
                    '400X': Image.open(img_400).convert('RGB')
                }
                
                # 2. Apply transforms and move to GPU/CPU
                tensor_dict = {
                    scale: transform(img).unsqueeze(0).to(device) 
                    for scale, img in images.items()
                }
                
                # 3. Feed forward through the network
                with torch.no_grad():
                    outputs, attn_weights = model(tensor_dict)
                    prob = torch.sigmoid(outputs).item()
                    
                    prediction = "Malignant" if prob > 0.5 else "Benign"
                    confidence = prob if prob > 0.5 else (1 - prob)
                
                # 4. Display the Final Result
                st.markdown("---")
                st.subheader("Diagnostic Report")
                
                if prediction == "Malignant":
                    st.error(f"**Diagnosis:** {prediction} (Confidence: {confidence*100:.2f}%)")
                else:
                    st.success(f"**Diagnosis:** {prediction} (Confidence: {confidence*100:.2f}%)")
                    
                # 5. Explainability AI (XAI)
                st.markdown("#### Neural Network Attention Weights")
                st.caption("This shows which magnification the AI relied on the most to make its decision.")
                
                weights = attn_weights[0].cpu().numpy().flatten()
                scales = ['40X', '100X', '200X', '400X']
                
                for i, scale in enumerate(scales):
                    st.progress(float(weights[i]), text=f"{scale}: {weights[i]*100:.1f}%")

else:
    st.info("Please upload all 4 scan magnifications to unlock the diagnostic engine.")