FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

# Expose port (Render sets $PORT dynamically, defaults to 8000)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI application with Uvicorn on dynamic PORT
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
