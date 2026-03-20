import cv2

from vehicle_recognition.brand_recognizer import VehicleBrandRecognizer
from vehicle_recognition.detector import VehicleDetector


def test_step_by_step(image_path):
    # 1. Khởi tạo (Sửa lại đường dẫn model của bạn)
    detector = VehicleDetector()
    brand_rec = VehicleBrandRecognizer()

    # Đọc ảnh gốc
    frame = cv2.imread(image_path)
    if frame is None:
        print("Không tìm thấy ảnh!")
        return

    # --- BƯỚC 2: DETECT XE ---
    vehicle_bbox = detector.detect(frame)
    if vehicle_bbox:
        x1, y1, x2, y2 = map(int, vehicle_bbox)

        # --- BƯỚC 3: CẮT XE (CROP) ---
        vehicle_crop = frame[y1:y2, x1:x2]
        cv2.imshow("Vung Anh Xe Da Cut (Crop)", vehicle_crop)

        # --- BƯỚC 4: NHẬN DIỆN LOGO & SHOW KẾT QUẢ CUỐI ---
        # Chạy model nhận diện logo trên ảnh đã cắt
        brand_name, confidence = brand_rec.predict(vehicle_crop)

        print(f"Tên hãng: {brand_name} --- Độ tin cậy: {confidence}")
        cv2.waitKey(0)
    else:
        print("Không tìm thấy xe!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_step_by_step("media/test/brand_1.png")