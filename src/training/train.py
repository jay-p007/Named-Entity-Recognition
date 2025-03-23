import os
import pickle
from transformers import TrainingArguments, Trainer
from src.models.models import model, tokenizer, tokenized_datasets

training_args = TrainingArguments(
    output_dir="./ner_model", evaluation_strategy="epoch", save_strategy="epoch",
    learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16,
    num_train_epochs=3, weight_decay=0.01, logging_dir="./logs", logging_steps=10,
    save_total_limit=2, push_to_hub=False
)
trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"], tokenizer=tokenizer
)
trainer.train()

# Define the model save path
model_save_path = "Ner_project/src/models/ner_model.pkl"

# Ensure directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the trained model
with open(model_save_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved successfully at: {model_save_path}")
