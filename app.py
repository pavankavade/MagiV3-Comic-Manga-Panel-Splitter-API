import os
import io
import torch
import warnings
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import numpy as np
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

# Configuration for border and effects
DEFAULT_BORDER_WIDTH = 10
BORDER_COLOR = "black"
DEFAULT_CORNER_RADIUS = 15
DEFAULT_SHADOW_OFFSET = (5, 5)
DEFAULT_SHADOW_BLUR = 10

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

def create_rounded_rectangle_mask(width, height, radius):
    """Create a mask for rounded corners."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
    return mask

def add_shadow_to_image(image, offset=(5, 5), blur_radius=10, shadow_color='gray'):
    """Add shadow effect to an image."""
    # Create shadow
    shadow = Image.new('RGBA', 
                      (image.width + abs(offset[0]) + blur_radius * 2, 
                       image.height + abs(offset[1]) + blur_radius * 2), 
                      (0, 0, 0, 0))
    
    # Create shadow shape
    shadow_img = Image.new('RGBA', image.size, shadow_color + (128,))  # Semi-transparent shadow
    
    # Apply the same mask to shadow if image has transparency
    if image.mode == 'RGBA':
        shadow_img.putalpha(image.split()[-1])  # Use original alpha
    
    # Blur the shadow
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Position shadow
    shadow_pos = (blur_radius + max(0, offset[0]), blur_radius + max(0, offset[1]))
    shadow.paste(shadow_img, shadow_pos)
    
    # Position original image
    img_pos = (blur_radius - min(0, offset[0]), blur_radius - min(0, offset[1]))
    shadow.paste(image, img_pos, image if image.mode == 'RGBA' else None)
    
    return shadow

def add_curved_border_and_shadow(image, 
                                border_width=DEFAULT_BORDER_WIDTH, 
                                border_color=BORDER_COLOR,
                                corner_radius=DEFAULT_CORNER_RADIUS,
                                add_shadow=True,
                                shadow_offset=DEFAULT_SHADOW_OFFSET,
                                shadow_blur=DEFAULT_SHADOW_BLUR):
    """Add curved border and shadow to an image."""
    
    # Convert to RGBA for transparency support
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Add border first
    bordered_img = ImageOps.expand(image, border=border_width, fill=border_color)
    
    # Create rounded corners
    width, height = bordered_img.size
    mask = create_rounded_rectangle_mask(width, height, corner_radius)
    
    # Apply rounded corners
    rounded_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    rounded_img.paste(bordered_img, (0, 0))
    rounded_img.putalpha(mask)
    
    # Add shadow if requested
    if add_shadow:
        final_img = add_shadow_to_image(rounded_img, shadow_offset, shadow_blur)
        # Convert back to RGB with white background
        result = Image.new('RGB', final_img.size, 'white')
        result.paste(final_img, (0, 0), final_img)
        return result
    else:
        # Convert back to RGB with white background
        result = Image.new('RGB', rounded_img.size, 'white')
        result.paste(rounded_img, (0, 0), rounded_img)
        return result

def add_border_to_image(image, border_width=DEFAULT_BORDER_WIDTH, border_color=BORDER_COLOR):
    """Add a simple border around the image (legacy function)."""
    return ImageOps.expand(image, border=border_width, fill=border_color)

# ---------- API ----------
@app.post("/split_panels")
async def split_panels(
    file: UploadFile = File(...),
    add_border: bool = Query(False, description="Whether to add a border to each panel"),
    border_width: int = Query(DEFAULT_BORDER_WIDTH, description="Width of the border in pixels", ge=1, le=50),
    border_color: str = Query(BORDER_COLOR, description="Color of the border (e.g., 'red', 'blue', '#FF0000')"),
    curved_border: bool = Query(False, description="Whether to add curved borders"),
    corner_radius: int = Query(DEFAULT_CORNER_RADIUS, description="Corner radius for curved borders", ge=5, le=50),
    add_shadow: bool = Query(False, description="Whether to add shadow effect"),
    shadow_blur: int = Query(DEFAULT_SHADOW_BLUR, description="Shadow blur radius", ge=1, le=30)
):
    """Takes an image, returns all cropped panel images as a ZIP file. 
    Optionally adds borders, curved corners, and shadows."""
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

        # Build ZIP in memory
        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, "w") as zipf:
            for i, bbox in enumerate(panel_boxes):
                # Crop the panel
                panel_image = image.crop(bbox)
                
                # Add effects if requested
                if add_border or curved_border or add_shadow:
                    if curved_border or add_shadow:
                        panel_image = add_curved_border_and_shadow(
                            panel_image, 
                            border_width=border_width if add_border else 0,
                            border_color=border_color,
                            corner_radius=corner_radius,
                            add_shadow=add_shadow,
                            shadow_blur=shadow_blur
                        )
                    elif add_border:
                        panel_image = add_border_to_image(panel_image, border_width, border_color)
                
                # Save to ZIP
                img_bytes = io.BytesIO()
                panel_image.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                zipf.writestr(f"panel_{i+1}.png", img_bytes.read())
        zip_buf.seek(0)

        # Generate filename based on effects applied
        effects = []
        if add_border or curved_border:
            effects.append("borders")
        if curved_border:
            effects.append("curved")
        if add_shadow:
            effects.append("shadow")
        
        filename = f"panels_{'_'.join(effects)}.zip" if effects else "panels.zip"
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test_first_panel")
async def test_first_panel(
    file: UploadFile = File(...),
    add_border: bool = Query(False, description="Whether to add a border to the panel"),
    border_width: int = Query(DEFAULT_BORDER_WIDTH, description="Width of the border in pixels", ge=1, le=50),
    border_color: str = Query(BORDER_COLOR, description="Color of the border (e.g., 'red', 'blue', '#FF0000')"),
    curved_border: bool = Query(False, description="Whether to add curved borders"),
    corner_radius: int = Query(DEFAULT_CORNER_RADIUS, description="Corner radius for curved borders", ge=5, le=50),
    add_shadow: bool = Query(False, description="Whether to add shadow effect"),
    shadow_blur: int = Query(DEFAULT_SHADOW_BLUR, description="Shadow blur radius", ge=1, le=30)
):
    """Test endpoint that takes an image and returns only the first detected panel as a single image download. 
    Optionally adds borders, curved corners, and shadows."""
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

        # Get only the first panel
        first_bbox = panel_boxes[0]
        panel_image = image.crop(first_bbox)
        
        # Add effects if requested
        if add_border or curved_border or add_shadow:
            if curved_border or add_shadow:
                panel_image = add_curved_border_and_shadow(
                    panel_image, 
                    border_width=border_width if add_border else 0,
                    border_color=border_color,
                    corner_radius=corner_radius,
                    add_shadow=add_shadow,
                    shadow_blur=shadow_blur
                )
            elif add_border:
                panel_image = add_border_to_image(panel_image, border_width, border_color)
        
        # Save to memory buffer
        img_bytes = io.BytesIO()
        panel_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Generate filename based on effects applied
        effects = []
        if add_border or curved_border:
            effects.append("border")
        if curved_border:
            effects.append("curved")
        if add_shadow:
            effects.append("shadow")
        
        filename = f"first_panel_{'_'.join(effects)}.png" if effects else "first_panel.png"
        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    return {
        "loaded": MODEL is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "border_settings": {
            "default_border_width": DEFAULT_BORDER_WIDTH,
            "border_color": BORDER_COLOR,
            "add_border_default": False,
            "curved_border_settings": {
                "default_corner_radius": DEFAULT_CORNER_RADIUS,
                "curved_border_default": False
            },
            "shadow_settings": {
                "default_shadow_offset": DEFAULT_SHADOW_OFFSET,
                "default_shadow_blur": DEFAULT_SHADOW_BLUR,
                "add_shadow_default": False
            }
        }
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
