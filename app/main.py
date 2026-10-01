import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.registry import ModelRegistry


logger = logging.getLogger("hq-face-api.access")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class RequestTimingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if scope["path"] == "/health":
                return
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s %s %.1fms",
                scope["method"],
                scope["path"],
                status_code,
                elapsed_ms,
            )


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", BASE_DIR / "uploads"))


def image_extension(content: bytes) -> str:
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    if content[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".bin"


def save_received_image(content: bytes) -> Path:
    now = datetime.now()
    day_dir = UPLOADS_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{now.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}{image_extension(content)}"
    path.write_bytes(content)
    return path


registry = ModelRegistry(
    str(BASE_DIR / "config" / "models.yaml")
)

app = FastAPI(
    title="Hidraquim Face API",
    version="1.0.0",
)
app.add_middleware(RequestTimingMiddleware)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "defaultModel": registry.default_model,
        "models": registry.list_models(),
    }


@app.get("/api/v1/faces/models")
async def models():
    return {
        "defaultModel": registry.default_model,
        "models": registry.list_models(),
    }


@app.post("/api/v1/faces/embedding")
async def embedding(
    image: UploadFile = File(...),
    modelCode: str = Form(None),
):
    content = await image.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_IMAGE",
                "message": "La imagen está vacía.",
            },
        )

    try:
        saved_path = save_received_image(content)
        logger.info("imagen guardada %s (%s bytes)", saved_path, len(content))
    except OSError:
        logger.exception("no se pudo guardar la imagen recibida")

    try:
        engine = registry.get(modelCode)
        result = engine.extract_embedding(content)

        return {
            "success": True,
            **result,
        }

    except KeyError:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "MODEL_NOT_FOUND",
                "message": f"ModelCode no soportado: {modelCode}",
            },
        )

    except ValueError as ex:
        code = str(ex)

        messages = {
            "INVALID_IMAGE": "No fue posible decodificar la imagen.",
            "FACE_NOT_FOUND": "No se detectó ningún rostro.",
            "MULTIPLE_FACES": "Se detectó más de un rostro.",
            "INVALID_EMBEDDING": "No fue posible generar el embedding.",
        }

        raise HTTPException(
            status_code=422,
            detail={
                "code": code,
                "message": messages.get(code, "Error procesando el rostro."),
            },
        )
