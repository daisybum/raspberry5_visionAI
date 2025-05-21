# ────────────────────────────────────────────────────────
#  Raspberry Pi 5  Vision Processor  (LiteRT + libcamera)
# ────────────────────────────────────────────────────────
FROM arm64v8/python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# 1) 시스템 / 개발 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        # ── 빌드 도구 ──────────────────────────────────
        build-essential python3-dev git pkg-config cmake meson ninja-build \
        # ── 필수 라이브러리 (libcamera·OpenCV 의존) ──
        libboost-program-options-dev libboost-dev                    \
        libgnutls28-dev libssl-dev                                  \
        libdrm-dev libudev-dev libexif-dev libyaml-cpp-dev           \
        libjpeg-dev libpng-dev libtiff5-dev                          \
        libglib2.0-dev                                              \
        # ── 선택 기능(GStreamer·FFmpeg·Qt·BLAS) ───────────
        libgstreamer-plugins-base1.0-dev                            \
        libavcodec-dev libavformat-dev libswscale-dev               \
        qtbase5-dev libqt5core5a libqt5widgets5                     \
        libopenblas-base                                            \
    && rm -rf /var/lib/apt/lists/*

# ── (선택) libcamera 소스 빌드 ───────────────────────────
# 라즈베리파이 OS가 아닌 순수 Debian 기반일 경우 libcamera-apps
# 패키지가 없으므로, 필요한 경우 아래 블록을 주석 해제해 직접 빌드
#
# RUN git clone --depth 1 https://git.libcamera.org/libcamera/libcamera.git && \
#     meson setup build libcamera -Dpipelines=raspberrypi -Dcam=enabled && \
#     ninja -C build && ninja -C build install && ldconfig && \
#     rm -rf libcamera

# 2) python → python3 심볼릭 링크 통합
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3) pip 최신화 + Python 패키지
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy pillow redis tqdm pycocotools jinja2 pyyaml ply tflite-runtime

# 4) 애플리케이션
WORKDIR /app
COPY pseudo_processor.py .

# 5) 스레드 관련 환경 변수
ENV NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

ENTRYPOINT ["python", "-u", "pseudo_processor.py"]
