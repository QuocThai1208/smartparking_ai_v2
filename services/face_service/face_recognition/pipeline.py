from .detector import FaceDetector
from .preprocessor import FacePreprocessor
from .recognizer import FaceRecognizer


class FaceRecognitionPipeline:
    def __init__(self):
        # Khởi tạo các thành phần
        self.detector = FaceDetector()
        self.preprocessor = FacePreprocessor()
        self.recognizer = FaceRecognizer()

    def run(self, image):
        """
        Luồng xử lý: Image -> Detection -> Alignment -> Embedding
        """
        # 1. Phát hiện khuôn mặt
        detection = self.detector.detector(image)

        if not detection:
            raise ValueError("error", "Không nhận diện được khuôn mặt.")

        # 2. Tiền xử lý (Cắt và xoay thẳng khuôn mặt)
        pre_data = self.preprocessor.process(image, detection)

        # 3. Trích xuất đặc trưng (Embedding)
        aligned_face = pre_data["processed_face"]
        embedding = self.recognizer.extract_embedding(aligned_face)

        return {
            "embedding": embedding,
            "processed_face": pre_data["processed_face"]
        }

