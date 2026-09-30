from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.registry import ModelRegistry


BASE_DIR = Path(__file__).resolve().parent.parent

registry = ModelRegistry(
    str(BASE_DIR / "config" / "models.yaml")
)

app = FastAPI(
    title="Hidraquim Face API",
    version="1.0.0",
)


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
