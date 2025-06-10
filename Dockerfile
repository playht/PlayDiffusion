# Use official Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Git LFS setup
RUN git lfs install

# Install PlayDiffusion
RUN git clone https://github.com/hanifalisohag/PlayDiffusion /app/PlayDiffusion

# Set working dir inside repo
WORKDIR /app/PlayDiffusion

# Upgrade pip and install dependencies (including demo)
RUN pip install --upgrade pip && \
    pip install '.[demo]' && \
    pip install -U huggingface_hub

# Create HuggingFace cache mount path
ENV HF_HOME=/app/.cache/huggingface

# Expose default gradio port
EXPOSE 7860

# Set default run command
CMD ["python", "demo/gradio-demo.py"]
