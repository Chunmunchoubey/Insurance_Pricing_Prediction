FROM python:3.9-slim

WORKDIR /app

# Requirements copy aur install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code copy karo
COPY app.py .
COPY models/ models/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]