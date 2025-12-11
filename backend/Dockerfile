FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model files and src directory from parent
COPY ../data/models ./data/models
COPY ../src ./src
COPY ../config ./config

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]