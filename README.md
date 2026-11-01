# DigiNova Object Detector API

A service that serves a ResNet-50-based object detection model (class + bounding box)-trained in Colab-via FastAPI within a Docker container.
Classes: `airplanes`, `Faces`, `Motorbikes`. The Docker image is configured to run on ARM-based Macs (`linux/arm64`).

## Running the Service

The model file resulting from the training process must be located at `model/detector.pth`.

```bash
git clone <repo-url>
cd <repo-folder>
docker-compose up
```

The service runs at `http://localhost:7001`. Swagger interface: http://localhost:7001/docs

## Usage

```bash
curl -X POST http://localhost:7001/predict -F "file=@example.jpg"
```

Example Response:

```json
{
  "filename": "example.jpg",
  "label": "Motorbikes",
  "probability": 0.9987,
  "bbox": {"startX": 12, "startY": 30, "endX": 245, "endY": 160}
}
```

If an invalid file is uploaded, the service returns a `400` error code along with an explanation.
