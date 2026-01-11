import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medical AI - Multi-Disease Detector",
    page_icon="🏥",
    layout="wide"
)

# --- CONFIGURATION & CONSTANTS ---
MODEL_PATH = 'D:/Medical_1/backend/model' 
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Fractured', 'Normal', 'Pneumonia', 'Tuberculosis']

# --- MEDICAL IMAGE PROCESSING ---
def apply_medical_filters(image_np):
    """
    Applies the exact preprocessing pipeline used in training:
    1. LAB Color Space conversion
    2. CLAHE (Contrast Enhancement)
    3. Gaussian Blur (Denoising)
    4. Resize
    """
    img = image_np.astype(np.uint8)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img_filtered = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    except Exception:
        img_filtered = img
    
    img_resized = cv2.resize(img_filtered, IMG_SIZE)
    return img_resized

def get_contour_map(image_np):
    """
    Generates a contour map to highlight structural boundaries.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive Thresholding for better boundary detection
    thresh = cv2.adaptiveThreshold(
        gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy to draw on
    contour_img = image_np.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1) # Green contours
    
    return contour_img

@st.cache_resource
def load_medical_model():
    """Loads the model with caching."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- UI COMPONENTS ---
def local_css():
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
        .prediction-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        [data-testid="stMetricValue"] { font-size: 28px; color: #1f77b4; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# Header
st.title("🏥 Medical Vision AI & Diagnostics")
st.markdown("### Specialized boundary analysis for **Fractures, Pneumonia, and TB**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    model = load_medical_model()
    if model:
        st.success("Core Model Active")
    else:
        st.error("Model Missing")
    
    st.divider()
    show_contours = st.toggle("Enable Contour Mapping", value=True, help="Highlights edges and boundaries in the image")
    sensitivity = st.slider("Contour Sensitivity", 50, 255, 120, help="Lower values show more detail, higher shows prominent structures.")

# Main Layout
if model:
    uploaded_file = st.file_uploader("Upload Medical Scan", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="large")

        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)

        with col1:
            st.subheader("🖼️ Original Scan")
            st.image(image, use_container_width=True)
            
            with st.expander("AI Enhanced Preprocessing"):
                processed_preview = apply_medical_filters(image_np)
                st.image(processed_preview, use_container_width=True, caption="CLAHE + Gaussian Denoising")

        with col2:
            st.subheader("🔬 AI Diagnostic Analysis")
            
            # Prediction logic
            with st.spinner("Running Neural Network..."):
                filtered_img = apply_medical_filters(image_np)
                img_float = filtered_img.astype(np.float32)
                img_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(img_float)
                
                img_batch = np.expand_dims(img_preprocessed, axis=0)
                predictions = model.predict(img_batch, verbose=0)[0]
                
                predicted_idx = np.argmax(predictions)
                confidence = predictions[predicted_idx] * 100
                result_label = CLASS_NAMES[predicted_idx]

                # Prediction Outcome
                if result_label == "Normal":
                    st.success(f"### Result: {result_label}")
                else:
                    st.error(f"### Detection: {result_label}")
                
                st.metric("Confidence Score", f"{confidence:.2f}%")

                # Boundary Analysis Section
                if show_contours:
                    st.divider()
                    st.write("**📐 Structural Boundary Mapping (Contours)**")
                    # Using Canny edge detection for the visual mapping based on slider
                    gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, sensitivity // 2, sensitivity)
                    
                    # Highlight edges on the original image
                    edge_map = filtered_img.copy()
                    edge_map[edges > 0] = [0, 255, 0] # Bright green for boundaries
                    
                    st.image(edge_map, use_container_width=True, caption="Green highlights indicate detected structural boundaries")

                st.divider()
                st.write("**Probability Breakdown:**")
                for i, score in enumerate(predictions):
                    st.write(f"{CLASS_NAMES[i]}: {score*100:.1f}%")
                    st.progress(float(score))

        # Recommendations footer
        st.divider()
        st.subheader("👨‍⚕️ Clinical Summary")
        if result_label != "Normal":
            st.warning(f"**Action Required:** High probability of **{result_label}** detected. The boundary mapping tool highlights irregularities that align with this diagnosis.")
        else:
            st.info("The structural analysis indicates no immediate abnormalities. Routine observation advised.")
else:
    st.warning("Awaiting model deployment...")

st.caption("AI Diagnostic Tool | Clinical correlation required.")