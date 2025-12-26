#file : \brain-tumor-backend\main.py
from app.main import app

# Convenience entrypoint so `python main.py` still works during local dev.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
