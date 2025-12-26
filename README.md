# Brain Tumor Classifier API

FastAPI backend that serves a TensorFlow EfficientNet model for MRI brain tumor classification into: `glioma`, `meningioma`, `notumor`, and `pituitary`.

## Features
- Single `/predict` endpoint accepting image uploads.
- JSON response with predicted class, confidence, and per-class probabilities.
- Model loaded once at startup for faster inference.
- Optional static frontend served at `/frontend`.

## Requirements
- Python 3.10+ recommended.
- `brain_tumor_efficientnet.keras` must exist in the project root (already included).
- GPU is not required; CPU inference works but may be slower.

## Project layout
- `app/config.py` constants for image size, class names, model path.
- `app/model.py` model loading, image preprocessing, prediction helper.
- `app/schemas.py` Pydantic response schema.
- `app/main.py` FastAPI app with `/health`, `/predict`, and static `/frontend`.
- `main.py` convenience entrypoint that runs the API via `uvicorn`.
- `brain_tumor_efficientnet.keras` trained model file.
- `test_inference.py` optional local smoke-test script.
- `frontend/index.html` minimal browser UI.

## Configuration
All runtime constants live in `app/config.py`:
- `IMG_SIZE = (150, 150)` for resize before inference.
- `CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]`.
- `MODEL_PATH = "brain_tumor_efficientnet.keras"`.

If you change the model or input size, update `IMG_SIZE`, `CLASS_NAMES`, and the model file path accordingly.

## Model and preprocessing
`app/model.py` performs preprocessing consistent with training:
- Converts input to RGB.
- Resizes to `IMG_SIZE`.
- Converts to `float32`.
- Applies `tensorflow.keras.applications.efficientnet.preprocess_input`.

This matches EfficientNet preprocessing and replaces simple `/ 255.0` normalization.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Alternative entrypoint:
```powershell
python main.py
```

Open the docs at `http://127.0.0.1:8000/docs` and health at `http://127.0.0.1:8000/health`.

## API endpoints

### GET `/health`
Simple health check.

Response:
```json
{ "status": "ok" }
```

### POST `/predict`
Accepts a multipart form upload with a single `file` field containing an image.

Example request:
```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "accept: application/json" `
  -F "file=@C:\path\to\mri.jpg"
```
the frontend : "http://127.0.0.1:8000/frontend/"
Response shape:
```json
{
  "predicted_class": "glioma",
  "confidence": 0.9876,
  "probabilities": {
    "glioma": 0.9876,
    "meningioma": 0.0042,
    "notumor": 0.0011,
    "pituitary": 0.0071
  }
}
```

Error cases:
- `400` if the uploaded file is not an image.
- `400` if the image cannot be decoded.

## Frontend
A minimal HTML UI is served from the `frontend` folder:
- Visit `http://127.0.0.1:8000/frontend` after starting the API.
- The page posts to `/predict` and displays the JSON response.

## Testing and validation
Run a quick inference smoke test:
```powershell
.\.venv\Scripts\Activate.ps1
python test_inference.py
```

This loads the model and performs a dummy prediction to confirm TensorFlow and the model file are working.

## Deployment notes
- The API currently allows all CORS origins. Tighten `allow_origins` in `app/main.py` for production.
- For production, run via a process manager and consider disabling `--reload`.
- Ensure the model file remains alongside the app or update `MODEL_PATH` to point to its deployed location.
