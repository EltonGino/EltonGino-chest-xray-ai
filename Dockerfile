FROM python:3.12-slim

# System deps — libgl1 + libglib2.0-0 required by opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache — only re-runs if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api.py .
COPY src/ ./src/
COPY configs/ ./configs/

# Checkpoint is mounted at runtime — not baked into the image
VOLUME ["/app/checkpoints"]

EXPOSE 8000

CMD ["python", "api.py"]
