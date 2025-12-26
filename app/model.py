from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import tensorflow as tf
# 1. We import the EXACT same preprocessing function used in Colab
from tensorflow.keras.applications.efficientnet import preprocess_input

from .config import IMG_SIZE, CLASS_NAMES, MODEL_PATH

_model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    """Load and cache the TensorFlow model."""
    global _model
    if _model is None:
        model_path = Path(MODEL_PATH).resolve()
        print(f"Loading model from: {model_path}")
        _model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded!")
    return _model


def preprocess_image(image: Image.Image) -> np.ndarray:
    # 2. Ensure RGB and resize
    img = image.convert("RGB").resize(IMG_SIZE)
    
    # 3. Convert to array (Values are 0 to 255)
    arr = np.array(img, dtype=np.float32)
    
    # 4. Apply EfficientNet preprocessing (replaces / 255.0)
    # This ensures the math is identical to your training phase
    arr = preprocess_input(arr)
    
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def predict(image: Image.Image) -> Dict:
    model = load_model()
    x = preprocess_image(image)
    preds = model.predict(x)
    probs = preds[0]  # shape (4,)

    best_idx = int(np.argmax(probs))
    best_class = CLASS_NAMES[best_idx]
    confidence = float(probs[best_idx])

    return {
        "predicted_class": best_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }