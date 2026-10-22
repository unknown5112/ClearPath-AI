FROM python:3.10-slim

WORKDIR /app

# OpenCV ke background processes ke liye zaroori libraries
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Requirements copy aur install karna
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tera saara code copy karna
COPY . .

# Hugging Face hamesha Port 7860 par chalta hai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]