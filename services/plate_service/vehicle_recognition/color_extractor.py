"""
    Nhận diện màu sắc
"""

from ultralytics import YOLO

class VehicleColorExtractor:
    def __init__(self, model_path="weights/yolo11n_cls_color_v1.pt"):
        self.model = YOLO(model_path)

    def predict(self, vehicle_crop):
        """
        :param vehicle_crop: ảnh xe đã được cắt từ ảnh gốc
        :return: trả vể tên màu (White, Black, Red, Silver...)
        """
        results = self.model.predict(vehicle_crop, verbose=False)
        # Lấy class có xác suất cao nhất
        probs = results[0].probs
        color_name = results[0].names[probs.top1]
        confidence = float(probs.top1conf)

        return color_name, confidence