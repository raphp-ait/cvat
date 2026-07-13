# Dockerfile for CVAT Webhook Receiver
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY custom/webhook/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY custom/webhook/app.py .
COPY custom/webhook/window_seg_inference.py .
COPY pytorch/custom/window_seg/nuclio/window_inference_shared.py .

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose webhook port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the webhook receiver
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
