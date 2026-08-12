FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for RDKit and health checks
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8023
ENV PYTHONUNBUFFERED=1

EXPOSE 8023

CMD ["python", "main.py"]
