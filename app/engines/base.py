from abc import ABC, abstractmethod
from typing import Any, Dict


class FaceEngine(ABC):
    def __init__(self, model_code: str):
        self.model_code = model_code

    @abstractmethod
    def extract_embedding(self, image_bytes: bytes) -> Dict[str, Any]:
        raise NotImplementedError
