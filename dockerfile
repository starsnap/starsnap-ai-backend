# StarSnap AI Backend Dockerfile (GPU)
# CUDA runtime + cuDNN 포함 이미지 사용
# CUDA 12.8 + cuDNN 9.8 supports Blackwell GPUs (for example RTX 5080,
# compute capability 12.0). Pin the amd64 manifest so production rebuilds use
# the runtime that passed GPU inference validation.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정 (GPU 활성화)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
# OpenGL 및 가상 디스플레이는 InsightFace/OpenCV 동작에 필요
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    curl \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    # deepface 등 의존성으로 CPU onnxruntime가 같이 깔릴 수 있어 GPU 패키지로 강제 고정
    python -m pip uninstall -y onnxruntime || true && \
    python -m pip install --no-cache-dir --upgrade onnxruntime-gpu==1.23.2 && \
    python -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())"


# 소스 코드 복사 (requirements 설치 후에 해야 캐시 효율적)
COPY . .

# 포트 노출
EXPOSE 8000

# 배포 롤아웃이 장애를 빠르게 감지할 수 있도록 짧은 주기로 확인한다.
HEALTHCHECK --interval=30s --timeout=10s --start-period=2m --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 애플리케이션 실행
CMD ["python", "app.py"]
