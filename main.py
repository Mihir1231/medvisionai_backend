import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64
import json

# --- CONFIGURATION ---
# Path for Multi-Disease X-Ray Model (Normal, Fracture, Pneumonia, TB)
MODEL_PATH_XRAY = 'D:/Medical_1/backend/model' 

# Paths for Specialized Pneumonia CT Model (Normal, Pneumonia)
CONFIG_PATH_CT = 'D:/Medical_1/backend/model_1/config.json'
WEIGHTS_PATH_CT = 'D:/Medical_1/backend/model_1/model.weights.h5'

IMG_SIZE = (224, 224)

# Dedicated Class Names
CLASS_NAMES_XRAY = ['Fractured', 'Normal', 'Pneumonia', 'Tuberculosis']
CLASS_NAMES_CT = ['Normal', 'Pneumonia']

app = FastAPI(
    title="Unified Medical Vision AI API",
    description="Multi-model backend for X-Ray (Multi-Disease) and CT (Pneumonia) analysis."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL MODEL STATE ---
model_xray = None
model_ct = None

@app.on_event("startup")
def load_models():
    """
    Loads both models into memory on startup for zero-latency switching.
    """
    global model_xray, model_ct
    
    # 1. Load Multi-Disease X-Ray Model (SavedModel format)
    if os.path.exists(MODEL_PATH_XRAY):
        try:
            model_xray = tf.keras.models.load_model(MODEL_PATH_XRAY)
            print("--- X-Ray Model loaded successfully.")
        except Exception as e:
            print(f"Error loading X-Ray model: {e}")
    else:
        print(f"X-Ray Model not found at {MODEL_PATH_XRAY}")

    # 2. Load Specialized CT Model (Architecture + Weights)
    if os.path.exists(CONFIG_PATH_CT) and os.path.exists(WEIGHTS_PATH_CT):
        try:
            with open(CONFIG_PATH_CT, 'r') as f:
                model_config = json.load(f)
            model_ct = tf.keras.models.model_from_json(json.dumps(model_config))
            model_ct.load_weights(WEIGHTS_PATH_CT)
            print("--- CT Pneumonia Model loaded successfully.")
        except Exception as e:
            print(f"Error loading CT model: {e}")
    else:
        print(f"CT Model components missing at {CONFIG_PATH_CT}")

# --- IMAGE PROCESSING UTILITIES ---

def apply_medical_filters(image_np):
    """
    Applies LAB color space conversion, CLAHE contrast enhancement, 
    and Gaussian denoising to standardize medical scans.
    """
    img = image_np.astype(np.uint8)
    try:
        # Convert to LAB to process luminosity independently
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # Denoising
        img_filtered = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    except Exception:
        img_filtered = img
    
    return cv2.resize(img_filtered, IMG_SIZE)

def generate_visual_mapping(filtered_img, sensitivity=120):
    """
    Generates a bright green contour overlay to highlight structural boundaries.
    """
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    # Canny Edge detection
    edges = cv2.Canny(gray, sensitivity // 2, sensitivity)
    # Bold the lines for better UI visibility
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    overlay = filtered_img.copy()
    overlay[edges > 0] = [0, 255, 0] # Green channels
    
    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "models": {
            "xray_loaded": model_xray is not None,
            "ct_loaded": model_ct is not None
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    modality: str = Query("xray", description="Either 'xray' or 'ct'"),
    sensitivity: int = 120
):
    # Select target model and classes based on modality
    target_model = model_xray if modality == "xray" else model_ct
    target_classes = CLASS_NAMES_XRAY if modality == "xray" else CLASS_NAMES_CT

    if target_model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"The {modality.upper()} model is currently unavailable on the server."
        )

    # 1. Read Image
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        image_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Medical Filtering
    processed_img = apply_medical_filters(image_np)
    
    # 3. Model Inference
    img_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(processed_img.astype(np.float32))
    input_batch = np.expand_dims(img_preprocessed, axis=0)
    
    preds = target_model.predict(input_batch, verbose=0)[0]
    best_idx = np.argmax(preds)
    label = target_classes[best_idx]
    confidence = float(preds[best_idx])

    # 4. Generate Visual Analysis
    visual_base64 = generate_visual_mapping(processed_img, sensitivity)

    # 5. Build Response
    return {
        "prediction": {
            "label": label,
            "confidence": confidence,
            "is_normal": label == "Normal",
            "modality_analyzed": modality
        },
        "breakdown": {target_classes[i]: float(preds[i]) for i in range(len(target_classes))},
        "visual_analysis": visual_base64,
        "metadata": {
            "filename": file.filename,
            "engine": "MobileNetV3-Medical",
            "modality": modality.upper()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)