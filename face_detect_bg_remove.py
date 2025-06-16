from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from retinaface import RetinaFace
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import asyncio
import threading

app = FastAPI()

# Add CORS middleware to allow requests from any origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_THREADS = 4
semaphore = threading.Semaphore(MAX_THREADS)

def crop_face_with_margin(img, facial_area, margin_percent):
    x1, y1, x2, y2 = [int(coord) for coord in facial_area]
    width = x2 - x1
    height = y2 - y1
    left = int(width * margin_percent[0])
    top = int(height * margin_percent[1])
    right = int(width * margin_percent[2])
    bottom = int(height * margin_percent[3])
    x1_new = max(x1 - left, 0)
    y1_new = max(y1 - top, 0)
    x2_new = min(x2 + right, img.shape[1])
    y2_new = min(y2 + bottom, img.shape[0])
    return img[y1_new:y2_new, x1_new:x2_new]

def process_image_sync(image_data):
    try:
        Image.open(BytesIO(image_data))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    bg_removed = remove(image_data)
    np_arr = np.frombuffer(bg_removed, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image after background removal")
    if img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img
    faces = RetinaFace.detect_faces(img_bgr)
    if not faces:
        raise HTTPException(status_code=404, detail="No face detected")
    first_face = next(iter(faces.values()))
    facial_area = first_face.get("facial_area")
    margin_percent = (0.2, 0.2, 0.2, 0.2)
    face_crop = crop_face_with_margin(img, facial_area, margin_percent)
    resized_face = cv2.resize(face_crop, (640, 640), interpolation=cv2.INTER_LINEAR)
    is_success, buffer = cv2.imencode(".png", resized_face)
    if not is_success:
        raise HTTPException(status_code=500, detail="Image encoding failed")
    return buffer.tobytes()

@app.post("/process")
async def process_image(file: UploadFile = File(..., description="Image file (form-data)")):
    """
    Upload an image file as form-data (key: 'file').
    """
    acquired = semaphore.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=429, detail="Too many requests, try again later.")
    try:
        image_data = await file.read()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_image_sync, image_data)
        return StreamingResponse(BytesIO(result), media_type="image/png")
    finally:
        semaphore.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info")
