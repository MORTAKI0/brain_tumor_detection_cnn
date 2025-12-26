#file : \brain-tumor-backend\app\main.py
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .model import load_model, predict
from .schemas import PredictionResponse

app = FastAPI(
    title="Brain Tumor Classifier API",
    version="1.0.0",
    description="Classify brain MRI into glioma / meningioma / notumor / pituitary.",
)

# CORS (so a frontend can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the simple HTML frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.on_event("startup")
def startup_event():
    # Warm up model once at startup
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    result = predict(image)
    return result


# For `python -m app.main`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
