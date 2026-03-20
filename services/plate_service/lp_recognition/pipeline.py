from .detector import PlateDetector
from .preprocessor import PlatePreprocessor
from .recognizer import PlateRecognizer


class LicensePlatePipeline:
    def __init__(self):
        self.detector = PlateDetector()
        self.preprocessor = PlatePreprocessor()
        self.recognizer = PlateRecognizer()

    def run(self, frame):
        """
        Luồng xử lý: Nhận diện vùng biển số -> Cắt & Làm nét -> Đọc ký tự
        :param frame: Ảnh gốc (Numpy array)
        :return: 'plate_text': '30A12345'
        """

        # BƯỚC 1: Phát hiện vị trí các biển số trong ảnh
        detections = self.detector.detect(frame)

        if not detections:
            return None

        # Sắp xếp lấy detection tốt nhất (conf cao nhất)
        best_detection = max(detections, key=lambda x: x['conf'])

        # BƯỚC 2: Cắt vùng biển số và tiền xử lý (Làm nét, cân bằng sáng CLAHE)
        processed_list = self.preprocessor.process(frame, [best_detection])

        if not processed_list:
            return None

        target_data = processed_list[0]  # Lấy kết quả đầu tiên (và duy nhất)

        # BƯỚC 3: Nhận diện ký tự từng biển số đã cắt
        plate_text = self.recognizer.recognize(target_data["processed_plate"])

        # Nếu OCR không đọc được chữ, trả về None để Service biết ảnh lỗi
        if not plate_text or plate_text.strip() == "":
            return None

        return plate_text