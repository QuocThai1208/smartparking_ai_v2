"""
    Phát hiện vùng khuôn mặt
"""

from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, conf_threshold=0.7):
        # Khởi tạo mô hình nhận diện khuôn mặt
        self.app = FaceAnalysis(allowed_modules=['detection'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.conf_threshold = conf_threshold

    def detector(self, image):
        """
        :param image: ảnh đầu vào từ camera
        :return:
        """
        faces = self.app.get(image)

        # Lọc theo ngưỡng confidence
        valid_faces = [f for f in faces if f.det_score >= self.conf_threshold]

        if not valid_faces:
            return None

        # Tìm khuôn mặt có độ tin cậy (conf) cao nhất
        best_face = max(valid_faces, key=lambda x: x.det_score)

        return {
            'bbox': best_face.bbox.astype(int).tolist(),
            "conf": float(best_face.det_score),
            "landmarks": best_face.kps
        }
