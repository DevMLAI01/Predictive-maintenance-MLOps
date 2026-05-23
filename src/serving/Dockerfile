FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch (avoids ~700 MB CUDA libs)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining runtime deps — pin numpy to 1.x to match training environment
RUN pip install --no-cache-dir \
    "numpy>=1.26,<2.0" \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.23" \
    "google-cloud-storage>=2.10"

# Model architecture + server
COPY src/model/lstm.py /app/lstm.py
COPY src/serving/serve.py /app/serve.py

EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
