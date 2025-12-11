FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
COPY app.py .
COPY ../src ./src
COPY ../config ./config
COPY ../data/models ./data/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 5000

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2", "app:app"]