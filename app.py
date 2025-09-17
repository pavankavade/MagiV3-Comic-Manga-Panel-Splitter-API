import os
import io
import torch
import warnings
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from pyngrok import ngrok
from transformers import AutoModel, AutoProcessor
import uvicorn

# Suppress noisy warnings about GenerationMixin
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")

# FastAPI app
app = FastAPI(title="MagiV3 Comic/Manga Panel Splitter API")

# CORS (open for demo; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load model ----------
@app.on_event("startup")
def load_model():
    global MODEL, PROCESSOR
    try:
        print("Loading model and processor...")
        MODEL = AutoModel.from_pretrained(
            "ragavsachdeva/magiv3",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(DEVICE).eval()

        PROCESSOR = AutoProcessor.from_pretrained(
            "ragavsachdeva/magiv3",
            trust_remote_code=True
        )
        print("✅ Model and processor loaded")
    except Exception as e:
        print("❌ Failed to load model:", e)
        raise

# ---------- API ----------
@app.post("/split_panels")
async def split_panels(file: UploadFile = File(...)):
    """Takes an image, returns cropped panel images as PNG streams."""
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Load uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Detect panels
        with torch.no_grad():
            results = MODEL.predict_detections_and_associations([image], PROCESSOR)
        page_result = results[0]
        panel_boxes = page_result["panels"]

        if not panel_boxes:
            return JSONResponse({"message": "No panels detected"})

        # Return first panel as demo stream (could zip all)
        panel_images = []
        for i, bbox in enumerate(panel_boxes):
            panel_image = image.crop(bbox)
            buf = io.BytesIO()
            panel_image.save(buf, format="PNG")
            buf.seek(0)
            panel_images.append(buf)

        # Right now: return only the first panel (to keep it simple)
        return StreamingResponse(panel_images[0], media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    return {
        "loaded": MODEL is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }

# ---------- Run server ----------
if __name__ == "__main__":
    port = 8000
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    ngrok_domain = os.getenv("NGROK_DOMAIN")

    if not ngrok_token or not ngrok_domain:
        raise RuntimeError("Set NGROK_AUTH_TOKEN and NGROK_DOMAIN in env")

    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port, "http", domain=ngrok_domain).public_url
    print(f"Public URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=port)
