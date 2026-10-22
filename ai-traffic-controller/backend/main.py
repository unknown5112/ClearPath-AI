import modal
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Modal App aur Cloud Environment banayenge
app = modal.App("clearpath-ai-backend")

# 2. Cloud Server setup: Libraries install karna aur GitHub se best.pt model download karna
# Note: Cloud par libGL error na aaye, isliye 'opencv-python-headless' use karna zaroori hai
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "uvicorn", "ultralytics", "opencv-python-headless", "python-multipart")
    .apt_install("wget", "libgl1", "libglib2.0-0")
    .run_commands("wget https://raw.githubusercontent.com/unknown5112/ClearPath-AI/main/ai-traffic-controller/backend/best.pt")
)

# 3. Tera original FastAPI setup (Modal ke app se alag rakhne ke liye iska naam 'web_app' diya hai)
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sabhi websites ko allow karega
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained YOLOv8 model
model = YOLO("best.pt")

@web_app.get("/")
def read_root():
    return {"message": "ClearPath AI Backend is permanently live on Modal 🚀"}

@web_app.post("/detect")
async def detect_emergency(file: UploadFile = File(...)):
    # Read frame directly in memory (RAM) for low-latency processing
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"emergency_detected": False, "boxes": []}

    h, w, _ = frame.shape

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, verbose=False)

    boxes_list = []
    emergency_detected = False

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower()
        confidence = float(box.conf[0])

        # Verify if the detected object is an emergency vehicle above the confidence threshold
        if ("emergency" in class_name or "ambulance" in class_name or "fire" in class_name or "police" in class_name) and confidence > 0.68:
            emergency_detected = True
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Normalize coordinates (0 to 1) for responsive rendering across all screen sizes
            boxes_list.append({
                "x1": x1 / w,
                "y1": y1 / h,
                "x2": x2 / w,
                "y2": y2 / h,
                "confidence": round(confidence * 100),
                "label": "EMERGENCY VEHICLE"
            })

    if emergency_detected:
        print(f"🚨 Detected {len(boxes_list)} Emergency Vehicle(s)!")

    return {
        "emergency_detected": emergency_detected,
        "boxes": boxes_list
    }

# 4. Modal ko batana ki is FastAPI app ko cloud par run karna hai
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app