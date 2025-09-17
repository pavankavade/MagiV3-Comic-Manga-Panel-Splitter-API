import os
import io
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import uvicorn

# -------------------
# Config
# -------------------
PORT = int(os.getenv("PORT", 8000))
CUSTOM_DOMAIN = os.getenv("NGROK_DOMAIN", "unadvantageously-clypeate-blakely.ngrok-free.app")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# -------------------
# FastAPI App
# -------------------
app = FastAPI(title="MagiV3 Comic Panel Splitter API")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Load Model
# -------------------
print("Loading model and processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = AutoModelForCausalLM.from_pretrained(
        "ragavsachdeva/magiv3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        "ragavsachdeva/magiv3",
        trust_remote_code=True
    )
except Exception as e:
    print("❌ Failed to load model:", e)
    model, processor = None, None

# -------------------
# Endpoints
# -------------------
@app.get("/")
async def home():
    return {"status": "API is running"}

@app.post("/split_panels")
async def split_panels(image: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Preprocess
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode into panel(s)
    panel_images = processor.decode(outputs[0], output_type="pil")

    if isinstance(panel_images, list):
        return JSONResponse(content={"num_panels": len(panel_images)})
    else:
        buf = io.BytesIO()
        panel_images.save(buf, format="PN_
