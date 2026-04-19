from .brand_recognizer import VehicleBrandRecognizer
from .color_extractor import VehicleColorExtractor
from .detector import VehicleDetector
from .type_classifier import VehicleTypeClassifier


class VehicleAttributePipeline:
    def __init__(self):
        self.detector = VehicleDetector()
        self.type_cls = VehicleTypeClassifier()
        self.color_ext = VehicleColorExtractor()
        self.brand_rec = VehicleBrandRecognizer()

    def run(self, frame):
        """
        Quy trình: Detect Xe -> Crop Xe -> [Nhận diện Type, Color, Brand]
        """
        # 1. Phát hiện xe
        bbox = self.detector.detect(frame)
        if not bbox:
            return None

        # 2. Cắt ảnh xe (Crop)
        x1, y1, x2, y2 = map(int, bbox)
        # Đảm bảo tọa độ không vượt quá biên ảnh
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        vehicle_crop = frame[y1:y2, x1:x2]

        # Kiểm tra nếu vùng crop quá nhỏ hoặc lỗi
        if vehicle_crop.size == 0:
            return None

        # 3. Phân loại Loại xe (Type) - Model Classification
        v_type, type_conf = self.type_cls.predict(vehicle_crop)

        # 4. Trích xuất màu sắc (Color) - Model Classification
        v_color, color_conf = self.color_ext.predict(vehicle_crop)

        # 5. Nhận diện hãng xe (Brand) - Model Detection (Nhìn vào vùng crop xe)
        v_brand, brand_conf = self.brand_rec.predict(vehicle_crop)

        # 6. Tổng hợp kết quả
        return {
            "type": v_type or None,
            "color": v_color or None,
            "brand": v_brand or None,
        }, vehicle_crop
