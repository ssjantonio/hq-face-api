# Hidraquim Face API

Microservicio facial extensible basado en un registry de modelos.

## Endpoints

- `GET /health`
- `GET /api/v1/faces/models`
- `POST /api/v1/faces/embedding`

## Modelo inicial

`opencv_sface_2021dec`

Requiere colocar manualmente:

- `models/opencv_sface_2021dec/face_detection_yunet_2023mar.onnx`
- `models/opencv_sface_2021dec/face_recognition_sface_2021dec.onnx`

Los modelos ONNX no se incluyen en el repositorio.

## Build

```bash
docker compose build
```

## Run

```bash
docker compose up -d
```

## Health

```bash
curl http://127.0.0.1:8090/health
```
