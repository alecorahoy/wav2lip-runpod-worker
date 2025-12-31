FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git ffmpeg python3 python3-pip python3-dev build-essential wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Rudrabha/Wav2Lip.git /workspace/Wav2Lip
WORKDIR /workspace/Wav2Lip

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Wav2Lip sometimes lists "sklearn" which breaks pip now. Replace it with scikit-learn.
RUN sed -i 's/^sklearn$/scikit-learn/' requirements.txt || true

# Install a modern, compatible dependency set for Wav2Lip inference (Python 3.10 friendly).
# We avoid Wav2Lip's old requirements.txt because it often fails to build.
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    scipy \
    librosa \
    numba \
    resampy \
    soundfile \
    tqdm \
    pillow \
    scikit-image \
    opencv-python-headless
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

RUN mkdir -p checkpoints && \
    echo "Downloading wav2lip_gan.pth from Hugging Face..." && \
    curl -L --retry 10 --retry-delay 5 --fail \
    -o checkpoints/wav2lip_gan.pth \
    https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth

COPY handler.py /workspace/Wav2Lip/handler.py

CMD ["python3", "-u", "handler.py"]
