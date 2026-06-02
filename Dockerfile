# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent writing pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV (libGL) and compiling dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the application files to the container
COPY . .

# Expose port 8501 for Streamlit access
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]
