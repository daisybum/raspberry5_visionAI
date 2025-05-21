FROM arm64v8/python:3.9-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Raspberry Pi camera dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcamera-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy~=1.24 \
    pillow~=10.0 \
    redis~=5.0

# Install LiteRT (TensorFlow Lite Runtime optimized for ARM64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxnnpack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow Lite Runtime with LiteRT optimizations
RUN pip install --no-cache-dir \
    https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl

WORKDIR /app
COPY vision_processor.py .

# Set environment variables for optimization
ENV NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

ENTRYPOINT ["python", "-u", "vision_processor.py"]
