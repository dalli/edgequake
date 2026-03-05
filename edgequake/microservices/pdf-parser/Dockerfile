# Use an official PyTorch runtime as a parent image, which includes CUDA Toolkit
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies required by docling and vllm
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install python dependencies. 
# We use --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Pre-download Qwen2-VL model weights? (Optional, reduces boot time but increases container build time/size)
# ENV VLM_MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# RUN python3 -c "from transformers import AutoProcessor, AutoModelForVision2Seq; AutoProcessor.from_pretrained('$VLM_MODEL_NAME'); AutoModelForVision2Seq.from_pretrained('$VLM_MODEL_NAME')"

# Command to run the FastApi application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
