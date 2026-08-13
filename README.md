# Insurance Pricing Prediction

End‑to‑end Machine Learning project for insurance premium prediction, featuring data preprocessing, model training, and FastAPI deployment with Docker & AWS.


## Overview

XGBoost-based insurance pricing prediction model with FastAPI and Docker.


## Dataset

[https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)


## Tech Stack

- Python, scikit-learn, XGBoost  
- FastAPI, Uvicorn  
- Docker  
- AWS (ECR, Lambda – if used in your setup)


## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train model:
   ```bash
   python train.py
   ```

3. Run API:
   ```bash
   python app.py
   ```

4. Test API at: http://localhost:8000/docs


## Docker

```bash
docker build -t insurance-pricing .
docker run -p 8000:8000 insurance-pricing
```


## Model Performance

- RMSE: ~$1,200  
- R²: ~0.85