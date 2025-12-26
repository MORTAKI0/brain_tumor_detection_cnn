#file : \brain-tumor-backend\test_inference.py
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.config import IMG_SIZE, MODEL_PATH
from app.model import load_model


def main():
    model_path = Path(MODEL_PATH)

    print("TensorFlow version:", tf.__version__)
    print("Loading model from:", model_path.resolve())

    model = load_model()
    print("バ. Model loaded!")

    # optional: run a dummy prediction to be sure
    dummy = np.zeros((*IMG_SIZE, 3), dtype=np.float32) / 255.0  # normalized like API
    dummy = np.expand_dims(dummy, axis=0)  # add batch dimension

    pred = model.predict(dummy)
    print("Prediction shape:", pred.shape)
    print("Raw prediction:", pred)


if __name__ == "__main__":
    main()
