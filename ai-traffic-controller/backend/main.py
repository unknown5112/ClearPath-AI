from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Vercel aur kisi bhi frontend ko backend se baat karne ki permission
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sabhi websites ko allow karega
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apna train kiya hua model load kar
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "AI Vision Backend is Running on Render!"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Image read karna
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # YOLO AI se prediction karwana
        results = model(img)
        
        # Detections ko JSON format mein pack karna
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                
                detections.append({
                    "class_name": name,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2]
                })

        return JSONResponse(content={"detections": detections})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)