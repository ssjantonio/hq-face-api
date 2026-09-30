import cv2
import numpy as np

from app.engines.base import FaceEngine


class OpenCvSFaceEngine(FaceEngine):
    def __init__(
        self,
        model_code: str,
        detector_path: str,
        recognizer_path: str,
        detection_threshold: float = 0.90,
    ):
        super().__init__(model_code)

        self.detector = cv2.FaceDetectorYN.create(
            detector_path,
            "",
            (320, 320),
            detection_threshold,
            0.3,
            5000,
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            recognizer_path,
            "",
        )

    def extract_embedding(self, image_bytes: bytes) -> dict:
        image_data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("INVALID_IMAGE")

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            raise ValueError("FACE_NOT_FOUND")

        if len(faces) > 1:
            raise ValueError("MULTIPLE_FACES")

        face = faces[0]
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned)

        embedding = feature.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)

        if norm <= 0:
            raise ValueError("INVALID_EMBEDDING")

        embedding /= norm

        confidence = float(face[-1])

        return {
            "modelCode": self.model_code,
            "embeddingDimensions": int(len(embedding)),
            "embedding": embedding.tolist(),
            "qualityScore": confidence,
            "boundingBox": {
                "x": int(round(float(face[0]))),
                "y": int(round(float(face[1]))),
                "width": int(round(float(face[2]))),
                "height": int(round(float(face[3]))),
            },
        }
