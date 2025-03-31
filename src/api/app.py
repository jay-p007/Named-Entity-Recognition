from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
import uvicorn
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Define model paths
LOCAL_MODEL_PATH = "Ner_project/src/training/ner_model"
REMOTE_MODEL_PATH = "Jay-007/Ner_model"

# Check if the local model path exists
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = REMOTE_MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Define NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

# Initialize FastAPI app
app = FastAPI(title="NER API", description="Named Entity Recognition with Fine-tuned XLM-RoBERTa", version="1.0")

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "NER API is running. Use /predict/ to make predictions."}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
def predict_entities(input_text: TextInput):
    text = input_text.text
    entities = ner_pipeline(text)

    print(entities)  # Debugging: Check entity output format

    # Convert model labels to meaningful names using model.config.id2label
    for entity in entities:
        entity["score"] = float(entity["score"])
        entity_label = entity["entity_group"]  # ✅ Use "entity_group" instead of "entity"

        # Get label from model.config.id2label, if available, otherwise keep original
        entity_id = entity_label.split("_")[-1]  # Extract ID from label (e.g., "LABEL_3" -> "3")
        try:
            entity_id = int(entity_id)
            entity["entity_group"] = model.config.id2label.get(entity_id, entity_label)
        except ValueError:
            entity["entity_group"] = entity_label  # Keep original if ID extraction fails

    return {"entities": entities}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
