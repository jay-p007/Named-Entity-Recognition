
## 1. Data Preprocessing & Feature Engineering

### Steps Taken:

- **Dataset Loading**: The dataset was loaded using the Hugging Face `datasets` library.
- **Tokenizer Selection**: Used the `XLM-RoBERTa tokenizer` for tokenizing input text.
- **Label Mapping**: Created mapping dictionaries (`label_to_id`, `id_to_label`) to convert named entity labels into numerical values.
- **Tokenization & Alignment**: Implemented tokenization while maintaining label alignment.
- **Class Weight Calculation**: Used inverse frequency to compute class weights for handling imbalance.

---

## 2. Model Selection & Optimization Approach

### Model Selection:

- **Pre-trained Model**: `xlm-roberta-large-finetuned-conll03-english` was used for token classification.
- **Custom Loss Functions**:
  - `WeightedCrossEntropyLoss` for addressing class imbalance.
  - `FocalLoss` to focus on difficult samples.
  - `MoMLoss` to improve `O`-class performance.
- **Training Configuration**:
  - Hugging Face `Trainer` API used for fine-tuning.
  - Hyperparameters optimized: `learning_rate=2e-5`, `num_train_epochs=3`, `batch_size=16`.
  - Models saved after each epoch.

### Training & Evaluation Metrics:

| Epoch | Training Loss | Validation Loss | Precision | Recall   | F1       | Accuracy |
| ----- | ------------- | --------------- | --------- | -------- | -------- | -------- |
| 1     | 0.007900      | 0.055435        | 0.954934  | 0.960404 | 0.957661 | 0.991914 |
| 2     | 0.001000      | 0.047132        | 0.956943  | 0.966133 | 0.961516 | 0.993103 |
| 3     | 0.000500      | 0.046741        | 0.961731  | 0.969671 | 0.965685 | 0.993707 |

Final Training Output:

```python
TrainOutput(global_step=2634, training_loss=0.017399892893629736, metrics={
    'train_runtime': 3846.0948,
    'train_samples_per_second': 10.952,
    'train_steps_per_second': 0.685,
    'total_flos': 9780213271931136.0,
    'train_loss': 0.017399892893629736,
    'epoch': 3.0
})
```

---

## 3. Deployment Strategy & API Usage Guide

### Deployment Steps:

- **Model Export**: The trained model was saved locally (`ner_model`) and remotely (`Jay-007/Ner_model` on Hugging Face Hub).
- **FastAPI-based API**:
  - Implemented an API using FastAPI for serving predictions.
  - Loaded the trained model dynamically.
  - Used `pipeline` for NER inference.
- **Streamlit Frontend**:
  - Developed a simple UI with Streamlit for interactive entity recognition.
  - Integrated frontend with FastAPI backend for real-time predictions.
- **Dockerization & Cloud Deployment**:
  - Packaged the application into a Docker container.
  - Deployed on Render for scalable API hosting.

### API Usage Guide:

- **Endpoint**: `/predict/`
- **Request Format**:
  ```json
  {
    "text": "Google was founded by Larry Page and Sergey Brin in California."
  }
  ```
- **Response Format**:
  ```json
  {
    "entities": [
      { "word": "Google", "entity_group": "ORG", "score": 1.00 },
      { "word": "Larry Page", "entity_group": "PER", "score": 1.00 },
      { "word": "Sergey Brin", "entity_group": "PER", "score": 1.00 },
      { "word": "California", "entity_group": "LOC", "score": 1.00 }
    ]
  }
  ```

---

