# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc libssl-dev

# Copy requirements files for the container
COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-test.txt .

# Copy the entire project into the container
COPY . .

# Install main requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install development requirements
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install test requirements
RUN pip install --no-cache-dir -r requirements-test.txt

# Run the application inside the container
CMD uvicorn challenge.api:app --host 0.0.0.0 --port $PORT