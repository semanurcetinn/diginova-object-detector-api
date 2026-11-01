from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model import load_detector, predict

app = FastAPI(title="DigiNova Object Detector API")
detector = load_detector()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        label, probability, bbox = predict(detector, image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {
        "filename": file.filename,
        "label": label,
        "probability": round(probability, 4),
        "bbox": bbox,
    }
