from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import cv2
import numpy as np
import  traceback

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Đang khởi tạo Face Pipeline... ---")
    try:
        from face_recognition.pipeline import FaceRecognitionPipeline
        models["face_pipeline"] = FaceRecognitionPipeline()
        print("--- Khởi tạo Model thành công! ---")
    except Exception as e:
        print("❌ LỖI KHỞI TẠO CHI TIẾT:")
        traceback.print_exc()

    yield

    # --- Code chạy khi SHUTDOWN ---
    print("--- Đang giải phóng tài nguyên... ---")
    models.clear()
    print("--- Đã đóng Server và giải phóng RAM ---")


app = FastAPI(lifespan=lifespan)


# Đọc ảnh
def to_cv2(file):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.post("/api/v1/extract-face")
async def extract_face(image_face: UploadFile = File(...)):
    if "face_pipeline" not in models:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    try:
        img = to_cv2(image_face)

        # Chạy pipeline từ dict models
        results = models["face_pipeline"].run(img)

        return {
            "success": True,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
