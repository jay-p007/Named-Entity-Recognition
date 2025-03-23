# Named Entity Recognition (NER) API

NER API is an AI-powered Named Entity Recognition (NER) system that enables users to extract named entities from text using a fine-tuned `xlm-roberta-large` model. The project includes a FastAPI-based backend, a Streamlit UI, and cloud deployment on Render.

---

## 📌 Features

✅ **Named Entity Recognition** for text input.

✅ **Pre-trained Transformer Model fine-tuned** on conll03-english dataset.

✅ **FastAPI-based API** for serving predictions.

✅ **Interactive UI** with Streamlit.

✅ **Dockerized Deployment** for cloud hosting.

---

## 🛠 Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.10+
- pip (Python package manager)
- Git (optional, for cloning the repository)
- Docker (optional, for containerization)

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/jay-p007/Named-Entity-Recognition.git
cd Named-Entity-Recognition
```

### 2️⃣ Create and Activate a Virtual Environment

```sh
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

---

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


## 🎯 Model Training & Evaluation

The model is fine-tuned using `xlm-roberta-large-finetuned-conll03-english`.

### 📊 Training & Evaluation Metrics:

| Epoch | Training Loss | Validation Loss | Precision | Recall   | F1 Score | Accuracy |
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

## 🌐 API Usage Guide

### 1️⃣ Running FastAPI Locally

```sh
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

This starts the API at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### 2️⃣ Running Streamlit UI Locally

```sh
streamlit run src/frontend/streamlit_app.py
```

This launches the UI in a web browser for interactive entity recognition.

### 3️⃣ API Request Example (Using Python)

```python
import requests

url = "http://127.0.0.1:8000/predict/"
data = {"text": "Google was founded by Larry Page and Sergey Brin in California."}

response = requests.post(url, json=data)
print(response.json())
```

### 4️⃣ Expected API Response Format

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

## 🚀 Deployment on Render

### Steps to Deploy FastAPI on Render:

1️⃣ Push your FastAPI app to a public GitHub repository.
2️⃣ Go to **[Render](https://render.com/)** and create a new Web Service.
3️⃣ Select your GitHub repository and set the **Start Command**:

```sh
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

4️⃣ Deploy and obtain the public API URL.

---

## 📂 Project Structure

```sh
Named-Entity-Recognition/
│-- src/
│   ├── data/
│   │   ├── dataset_loader.py  # Dataset loading script
│   │   ├── processing.py  # Data preprocessing
│   ├── api/
│   │   ├── app.py  # FastAPI Application  
│   ├── frontend/
│   │   ├── streamlit_app.py  # Streamlit UI
│   ├── models/
│   │   ├── loss_functions.py  # Custom Loss Functions
│   │   ├── models.py  # Model Loading
│   ├── training/
│   │   ├── train.py  # Training Script
│-- requirements.txt  # Dependencies
│-- Dockerfile  # Containerization
│-- README.md  # Documentation
```

---

## 📌 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

Enjoy using the NER API! 🚀🎉

