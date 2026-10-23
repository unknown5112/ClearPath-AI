# 🚨 ClearPath AI: Smart Traffic Command Center

## 📌 Overview
**ClearPath AI** is a real-time computer vision system designed to optimize urban traffic flow for emergency vehicles. By detecting Ambulances, Fire Trucks, and Police cars in real-time, it acts as the brain for adaptive traffic signals, ensuring clear pathways for first responders and reducing critical response times.

***🚀 Live Demo:** https://clearpath-ai-frontend.vercel.app*

## 🛠️ Tech Stack & Architecture

* **Frontend:** HTML, CSS (Tailwind), JavaScript, hosted on **Vercel**.
* **AI/ML Engine:** Custom-configured **YOLOv8** (`best.pt`) for high-precision object detection.
* **Backend API:** **FastAPI** handling concurrent asynchronous requests.
* **Image Processing:** **OpenCV** & **NumPy** for low-latency, in-memory frame decoding (`cv2.imdecode`).
* **Cloud Infrastructure:** Serverless container deployment on **Modal** (2GB RAM, headless Debian Linux) for scaling deep learning workloads without OOM crashes.

## ✨ Key Features
* **Real-Time Emergency Detection:** Exclusively filters for critical vehicles with a strict `> 68%` confidence threshold to prevent false positives.
* **Zero-Lag Processing:** Bypasses disk I/O by reading image byte streams directly into RAM (`np.frombuffer`).
* **Responsive Coordinate Normalization:** Bounding box coordinates are normalized (`0 to 1`) at the API level, ensuring flawless rendering on any device or screen size.
* **Serverless Scalability:** The Modal cloud architecture dynamically spins up resources to handle YOLOv8 inference, scaling down to zero when inactive.

## ⚙️ How It Works (The Pipeline)
1. **Client Request:** The user uploads a traffic snapshot via the Vercel web interface.
2. **Serverless Trigger:** The FastAPI endpoint on Modal wakes up and receives the byte payload.
3. **In-Memory Decoding:** OpenCV reconstructs the image in RAM.
4. **YOLOv8 Inference:** The deep learning model scans the frame for emergency vehicles.
5. **JSON Response:** The backend returns a boolean flag (`emergency_detected`) and normalized bounding box coordinates.
6. **UI Rendering:** The frontend dynamically draws targeted overlays over the detected vehicles.

## 🚀 Run Locally

### Prerequisites
* Python 3.10+
* Git

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/unknown5112/ClearPath-AI.git
   cd ClearPath-AI
   ```
2. **Setup the Backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:web_app --reload
   ```
3. **Setup the Frontend:**
   * Open `frontend/index.html` in your browser or run a simple local server using Live Server in VS Code.
   * *Note: Ensure the API endpoint in `index.html` points to your active backend (Localhost or Modal URL).*

## 🔮 Future Scope
* **Hardware Integration:** Connect the cloud API with a 4-way density-based traffic signal controller using an **Arduino Uno**, servo motors, and IR sensors for a complete cyber-physical system.
* **Video Stream Support:** Upgrade the FastAPI endpoint to process live CCTV RTSP feeds using WebSocket connections.

## 🤝 Connect with the Developer
Built with ☕ and late-night debugging. 
Let's connect and discuss AI, Cloud Architecture, and scalable systems!

*[LinkedIn Profile](https://www.linkedin.com/in/vishesh-choudhary-737437441?utm_source=share_via&utm_content=profile&utm_medium=member_android)*
