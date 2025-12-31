FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git ffmpeg python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Rudrabha/Wav2Lip.git /workspace/Wav2Lip
WORKDIR /workspace/Wav2Lip

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

RUN mkdir -p checkpoints && \
    wget -O checkpoints/wav2lip_gan.pth \
    https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth

COPY handler.py /workspace/Wav2Lip/handler.py

CMD ["python3", "-u", "handler.py"]
