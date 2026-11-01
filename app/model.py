from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50

# During training, LabelEncoder sorts the classes alphabetically, which is why the class order is as follows.
CLASSES = ["Faces", "Motorbikes", "airplanes"]
IMAGE_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "detector.pth"


class ObjectDetector(nn.Module):
    # Exactly the same model I trained in Colab: ResNet-50 + box head + class head.
    def __init__(self, base_model: nn.Module, num_classes: int):
        super().__init__()
        self.base_model = base_model
        in_features = base_model.fc.in_features

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        return self.regressor(features), self.classifier(features)


def load_detector() -> ObjectDetector:
    # Sets up the architecture, loads the saved weights, and switches to eval mode.
    model = ObjectDetector(resnet50(weights=None), len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


def predict(model: ObjectDetector, image_bytes: bytes):
    # Image bytes (label, probability, bounding box in pixels). Preprocessing is the same as for training.
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Yüklenen dosya geçerli bir görüntü değil.")
    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = ((tensor - MEAN) / STD).unsqueeze(0)

    with torch.inference_mode():
        box, logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    idx = int(probs.argmax())
    sx, sy, ex, ey = box[0].tolist()
    pixel_box = {"startX": int(sx * w), "startY": int(sy * h), "endX": int(ex * w), "endY": int(ey * h)}
    return CLASSES[idx], float(probs[idx]), pixel_box
