# Named Entity Recognition (NER) API

## Project Overview
This project trains an **XLM-Roberta-based Named Entity Recognition (NER) model** and deploys it via a FastAPI REST service. The dataset is **imbalanced**, so we improve model performance using:
- **Weighted Cross-Entropy (WCE) Loss**
- **Focal Loss**
- **Majority or Minority (MoM) Loss**

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ner-project.git
   cd ner-project
