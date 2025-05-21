###############################################################################
#  Raspberry Pi 5  Vision Processor – Slim Runtime Edition (Bookworm, arm64)
###############################################################################
# ──────────────── 1) 빌드 스테이지 ────────────────
FROM arm64v8/python:3.11-slim-bookworm AS builder
ENV DEBIAN_FRONTEND=noninteractive

# ── 공통 레포지토리(GPG) 추가 ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates apt-transport-https

# ── Raspberry Pi(bookworm) ────────────────────────
RUN curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] "\
        "http://archive.raspberrypi.org/debian bookworm main" \
        > /etc/apt/sources.list.d/raspi.list

# ── libcamera 빌드 의존 패키지 ────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git meson ninja-build cmake pkg-config \
        pybind11-dev qtbase5-dev \
        libboost-program-options-dev libgnutls28-dev openssl \
        libdrm-dev libudev-dev libexif-dev libyaml-cpp-dev libglib2.0-dev \
        libjpeg-dev libpng-dev libtiff5-dev \
        libgstreamer-plugins-base1.0-dev libavcodec-dev libavformat-dev libswscale-dev \
        python3-pip

# ── libcamera 소스 빌드 & 설치 ───────────────────
RUN git clone --depth 1 https://git.libcamera.org/libcamera/libcamera.git /src/libcamera \
 && sed -i '/removeprefix/s/\(.*\)= kebab_case(\(.*\)\.removeprefix(\(.*\)))$/\1 = kebab_case((\2[len(\3):] if \2.startswith(\3) else \2))/' \
        /src/libcamera/utils/codegen/gen-gst-controls.py \
 && meson setup /src/libcamera/build /src/libcamera \
 && ninja -C /src/libcamera/build && ninja -C /src/libcamera/build install \
 && echo "/usr/local/lib" > /etc/ld.so.conf.d/libcamera.conf \
 && ldconfig

# ──────────────── 2) 런타임 스테이지 ──────────────
FROM arm64v8/python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# ── 레포지토리(GPG) 동일하게 추가 ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates apt-transport-https \
 && curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] "\
        "http://archive.raspberrypi.org/debian bookworm main" \
        > /etc/apt/sources.list.d/raspi.list \
 && apt-get update

# ── 필수 런타임 패키지 ────────────────────────────
RUN apt-get install -y --no-install-recommends \
        libedgetpu1-std python3-pycoral python3-tflite-runtime \
        libcamera-apps python3-picamera2 \
        libqt5core5a libqt5widgets5 libgstreamer-plugins-base1.0-0 \
        libjpeg-turbo8 libpng16-16 libtiff5 libdrm2 libudev1 libglib2.0-0 libexif12 \
 && rm -rf /var/lib/apt/lists/*

# ── 빌드 스테이지에서 컴파일된 libcamera 복사 ───── 
COPY --from=builder /usr/local /usr/local

# ── python ↔ python3 심볼릭 링크 ────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ── 경량 파이썬 패키지 설치 ───────────────────────
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel \
 && pip install --no-cache-dir pillow tqdm pyyaml

# ── 애플리케이션 파일 복사 ────────────────────────
WORKDIR /app
COPY pseudo_processor.py .

# ── 기본 ENTRYPOINT (디버깅용) ─────────────────────
ENTRYPOINT ["tail", "-f", "/dev/null"]
# 본래 실행:  ENTRYPOINT ["python", "-u", "pseudo_processor.py"]
