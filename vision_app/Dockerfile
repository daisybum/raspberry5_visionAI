###############################################################################
# Raspberry Pi 5 Vision Processor  –  Slim Runtime Edition
###############################################################################
FROM arm64v8/python:3.11-slim-bookworm

# ────────────── 전역 설정 ──────────────
ENV LANG=C.UTF-8 TZ=Asia/Seoul DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

# ────────────── 필수 패키지 ──────────────
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # 빌드 툴 (pycocotools 컴파일용)
        build-essential python3-dev git pkg-config \
        # 런타임 공유 라이브러리
        libjpeg62-turbo libpng16-16 libtiff6 \
        libglib2.0-0 libudev1 libdrm2 libexif12 libyaml-cpp0.7 \
        libopenblas0-pthread \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ────────────── Python 의존성 ──────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# ────────────── 애플리케이션 ──────────────
WORKDIR /app
COPY pseudo_processor.py .

ENTRYPOINT ["tail", "-f", "/dev/null"]
# ENTRYPOINT ["python", "-u", "pseudo_processor.py"]
