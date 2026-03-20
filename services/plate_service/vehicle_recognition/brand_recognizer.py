"""
    Detect logo, Nhận diện hãng xe
"""
from ultralytics import YOLO

# names: ['Audi', 'BYD', 'Chery', 'Ford', 'Foton', 'Honda', 'Hyundai', 'Isuzu', 'JAC', 'Jetour', 'Kia', 'Mazda', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Suzuki', 'Tesla', 'Toyota']


class VehicleBrandRecognizer:
    def __init__(self, model_path="weights\yolo11n_brands_v1.pt"):
        self.model = YOLO(model_path)

    def predict(self, vehicle_crop):
        """
        Input: Ảnh đã cắt vùng đầu xe (vehicle_crop)
        Output: Tên hãng xe (e.g., 'Toyota') hoặc 'Unknown' nếu không thấy logo
        """
        # Chạy inference
        results = self.model.predict(vehicle_crop, conf=0.4, verbose=False)

        result = results[0]

        # Kiểm tra xem có tìm thấy logo nào không
        if len(result.boxes) > 0:
            # Lấy logo có độ tự tin (confidence) cao nhất
            # YOLO mặc định sắp xếp boxes theo conf từ cao xuống thấp
            top_box = result.boxes[0]
            class_id = int(top_box.cls[0])
            brand_name = result.names[class_id]
            confidence = float(top_box.conf[0])

            return brand_name, confidence

        return "Unknown", 0.0
