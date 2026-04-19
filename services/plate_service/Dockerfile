FROM python:3.11.9-slim

# Cài đặt các thư viện hệ thống cần thiết cho AI (như OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào
COPY . .

# Chạy service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]