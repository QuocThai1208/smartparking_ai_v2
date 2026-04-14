import base64

import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Biến global chứa model
models = {}


# 2. SỬ DỤNG LIFESPAN THAY CHO ON_EVENT
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code chạy khi STARTUP ---
    print("--- Đang khởi tạo Model AI... ---")
    try:
        # Import lazy để tránh xung đột
        from lp_recognition.pipeline import LicensePlatePipeline
        from vehicle_recognition.pipeline import VehicleAttributePipeline
        # Lưu vào dict models để dùng chung
        models["plate_pipeline"] = LicensePlatePipeline()
        models["attribute_pipeline"] = VehicleAttributePipeline()

        print("--- Model AI đã sẵn sàng! ---")
    except Exception as e:
        print(f"Lỗi khởi tạo Model: {e}")

    yield  # Server bắt đầu nhận request tại đây

    # --- Code chạy khi SHUTDOWN ---
    print("--- Đang giải phóng tài nguyên... ---")
    models.clear()


app = FastAPI(title="SmartParking AI Service", lifespan=lifespan)


# Đọc ảnh
def to_cv2(file):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_base64(image):
    if image is None: return None
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


@app.post("/api/v1/predict-vehicle")
async def predict_vehicle(
        image_front: UploadFile = File(...),
        image_plate: UploadFile = File(...)
):
    if "plate_pipeline" not in models:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    try:
        img_f = to_cv2(image_front)
        img_p = to_cv2(image_plate)

        # Chạy dự đoán
        res_plate, processed_plate = models["plate_pipeline"].run(img_p)
        res_attr, vehicle_crop = models["attribute_pipeline"].run(img_f)

        return {
            "success": True,
            "data": {
                "plate": res_plate,
                "attributes": res_attr,
            },
            "file": {
                "processed_plate": cv2_to_base64(processed_plate),
                "vehicle_crop": cv2_to_base64(vehicle_crop)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
