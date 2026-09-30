from typing import Dict, Optional

import yaml

from app.engines.base import FaceEngine
from app.engines.opencv_sface import OpenCvSFaceEngine


class ModelRegistry:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.default_model = config["default_model"]
        self._engines: Dict[str, FaceEngine] = {}

        for model_code, model_config in config["models"].items():
            engine_type = model_config["engine"]

            if engine_type == "opencv_sface":
                engine = OpenCvSFaceEngine(
                    model_code=model_code,
                    detector_path=model_config["detector"],
                    recognizer_path=model_config["recognizer"],
                    detection_threshold=model_config.get(
                        "detection_threshold",
                        0.90,
                    ),
                )
            else:
                raise ValueError(f"Engine no soportado: {engine_type}")

            self._engines[model_code] = engine

        if self.default_model not in self._engines:
            raise ValueError(
                f"El modelo default '{self.default_model}' no está registrado."
            )

    def get(self, model_code: Optional[str] = None) -> FaceEngine:
        code = model_code or self.default_model

        if code not in self._engines:
            raise KeyError(code)

        return self._engines[code]

    def list_models(self):
        return list(self._engines.keys())
