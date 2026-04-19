"""
    Nhận diện loại xe
"""
from ultralytics import YOLO


class VehicleTypeClassifier:
    def __init__(self, model_path="weights/yolo11n_cls_type_v1.pt"):
        self.model = YOLO(model_path)

    def predict(self, vehicle_crop):
        """
        :param vehicle_crop: ảnh xe đã được cắt từ ảnh gốc
        :return: label: tên loại xe, confidence: độ tin cậy
        """
        results = self.model.predict(vehicle_crop, verbose=False)
        result = results[0]

        # Lấy tên class có xác suất cao nhất (e.g., "Sedan")
        probs = result.probs
        class_id = probs.top1
        label = result.names[class_id]
        confidence = float(probs.top1conf)

        return label.upper(), confidence