from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
import os

# Initialize FastAPI app
app = FastAPI()

# Model loading code remains the same
MODEL_PATH = "Jay-007/Ner_model"  # Use remote model path directly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, 
                          aggregation_strategy="simple",
                          device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading model: {e}")
    model, tokenizer, ner_pipeline = None, None, None

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict_entities(input_text: TextInput):
    if not ner_pipeline:
        return {"error": "Model not loaded"}
        
    try:
        entities = ner_pipeline(input_text.text)
        return {"entities": entities}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
