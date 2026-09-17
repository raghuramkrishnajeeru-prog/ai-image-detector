from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

processor = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
model = AutoModelForImageClassification.from_pretrained(
    "Ateeqq/ai-vs-human-image-detector"
)

@app.get("/")
def root():
    return {"status": "AI detection backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_class_id = probs.argmax().item()
    label = model.config.id2label[predicted_class_id]
    confidence = probs[predicted_class_id].item()

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }
