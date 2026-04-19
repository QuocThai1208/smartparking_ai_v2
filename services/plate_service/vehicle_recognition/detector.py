"""
    Tìm và cắt khung hình xe
"""
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="weights/yolo11n.pt"):
        # Load model YOLOv11 Nano (nhẹ nhất để chạy CPU)
        self.model = YOLO(model_path)
        # Chỉ quan tâm các class: 2 (car), 5 (bus), 7 (truck) theo COCO dataset
        self.target_classes = [2, 5, 7]

    def detect(self, frame):
        """
        :param frame: ảnh đầy vào từ camera
        :return: [x1, y1, x2, y2] tọa độ box của xe
        """
        results = self.model.predict(frame, conf=0.5, verbose=False)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) in self.target_classes:
                    # Trả về tọa độ [x1, y1, x2, y2] của xe đầu tiên tìm thấy
                    return box.xyxy[0].tolist()
        return None