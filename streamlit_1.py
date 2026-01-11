import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pneumonia AI Detector",
    page_icon="🫁",
    layout="wide"
)

# --- CONSTANTS ---
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Pneumonia'] 

# --- MODEL LOADING ---
@st.cache_resource
def load_pneumonia_model():
    """
    Loads the model using the configuration and weights files.
    """
    config_path = 'D:/Medical_1/backend/model_1/config.json'
    weights_path = 'D:/Medical_1/backend/model_1/model.weights.h5'
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        st.error("Model files (config.json or model.weights.h5) missing in the directory.")
        return None
    
    try:
        # Load architecture from JSON
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        model = tf.keras.models.model_from_json(json.dumps(model_config))
        # Load weights
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- IMAGE PROCESSING ---
def preprocess_image(image_np):
    """
    Standard preprocessing for medical imaging:
    1. Grayscale to RGB if needed
    2. Resize to 224x224
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for X-ray clarity
    4. MobileNetV3 specific scaling
    """
    # Ensure 3 channels
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Contrast Enhancement (Crucial for X-rays)
    img_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Resize
    img_resized = cv2.resize(img_enhanced, IMG_SIZE)
    
    # Scaling
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    
    return img_preprocessed, img_enhanced

def get_contour_map(image_np, sensitivity=120):
    """
    Generates a high-precision contour map for structural analysis.
    """
    # Convert to gray and denoise
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, sensitivity // 2, sensitivity)
    
    # Dilation to make boundaries more visible
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create overlay
    contour_img = image_np.copy()
    contour_img[edges > 0] = [0, 255, 0] # Bright Green
    
    return contour_img

# --- UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .confidence-text { font-size: 24px; font-weight: bold; color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    st.title("🫁 Pneumonia Detection AI")
    st.markdown("### Specialized Deep Learning Analysis for Chest X-Rays")
    
    model = load_pneumonia_model()

    with st.sidebar:
        st.header("System Status")
        if model:
            st.success("AI Model Loaded Successfully")
        else:
            st.error("Model Offline")
        
        st.divider()
        st.header("🔍 Analysis Tools")
        enable_contours = st.toggle("Enable Contour Mapping", value=True)
        contour_sensitivity = st.slider("Boundary Sensitivity", 50, 255, 120)
        
        st.divider()
        st.info("This tool uses a MobileNetV3 backbone trained specifically to differentiate between 'Normal' lungs and those showing signs of 'Pneumonia'.")

    if model:
        uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display layout
            col1, col2 = st.columns([1, 1])

            # Load and Preprocess
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            processed_img, enhanced_viz = preprocess_image(image_np)

            with col1:
                st.subheader("Input Scan")
                st.image(image, use_container_width=True, caption="Original Upload")
                
                if enable_contours:
                    with st.expander("Structural Boundary Map", expanded=True):
                        contour_viz = get_contour_map(enhanced_viz, contour_sensitivity)
                        st.image(contour_viz, use_container_width=True, caption="Green highlights indicate detected structural boundaries")

            with col2:
                st.subheader("AI Diagnostic Results")
                
                with st.spinner("Analyzing scan patterns..."):
                    # Prediction
                    batch = np.expand_dims(processed_img, axis=0)
                    preds = model.predict(batch, verbose=0)[0]
                    
                    # Result handling
                    res_idx = np.argmax(preds)
                    confidence = preds[res_idx] * 100
                    label = CLASS_NAMES[res_idx]

                    # Visual Feedback
                    if label == "Normal":
                        st.balloons()
                        st.success(f"### Result: {label}")
                    else:
                        st.error(f"### Result: {label}")

                    st.write(f"**Confidence Level:** {confidence:.2f}%")
                    st.progress(float(preds[res_idx]))

                    # Probability Breakdown
                    st.divider()
                    st.write("**Full Analysis:**")
                    for i, prob in enumerate(preds):
                        col_a, col_b = st.columns([1, 3])
                        col_a.text(CLASS_NAMES[i])
                        col_b.progress(float(prob))

            # Footer Clinical Note
            st.divider()
            if label == "Pneumonia":
                st.warning("**Clinical Warning:** The AI has detected features consistent with Pneumonia. Immediate radiologist review is recommended.")
            else:
                st.info("**Observation:** No significant signs of Pneumonia detected. Maintain clinical correlation with patient symptoms.")

    else:
        st.warning("Please ensure 'config.json' and 'model.weights.h5' are in the application folder.")

    st.caption("AI Diagnostic Assistant | Not a replacement for professional medical advice.")

if __name__ == "__main__":
    main()