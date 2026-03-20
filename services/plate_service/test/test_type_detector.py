import cv2

from vehicle_recognition.detector import VehicleDetector
from vehicle_recognition.type_classifier import VehicleTypeClassifier


def test_type_pipeline(image_path):
    # 1. Khởi tạo model
    detector = VehicleDetector()
    type_extractor = VehicleTypeClassifier()

    # 2. Đọc ảnh từ file
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Không tìm thấy ảnh tại: {image_path}")
        return

    # 3. Bước 1: Detect xe
    print("--- Bước 1: Đang tìm xe trong ảnh ---")
    vehicle_bbox = detector.detect(frame)

    if vehicle_bbox:
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        # 4. Bước 2: Cắt vùng ảnh xe (Crop)
        vehicle_crop = frame[y1:y2, x1:x2]

        # 5. Bước 3: Nhận diện loại xe
        print("--- Bước 2: Đang phân tích loại xe ---")
        type_name, confidence = type_extractor.predict(vehicle_crop)

        # 6. Hiển thị và lưu kết quả
        result_text = f"Type: {type_name} , ({confidence * 100:.1f}%)"
        print(f"🎯 Kết quả: {result_text}")

        # Show các bước
        cv2.imshow("1. Vung xe da Crop", vehicle_crop)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Không tìm thấy xe nào để nhận diện loại xe.")


if __name__ == "__main__":
    test_type_pipeline("media/test/brand_4.jpg")