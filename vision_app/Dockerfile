###############################################################################
# Raspberry Pi 5 Vision Processor – Slim Runtime Edition (Bookworm, arm64)
#  · Python 3.9 기반
#  · EdgeTPU C-런타임(libedgetpu) + TFLite-runtime만 포함
###############################################################################
########## 1) 빌드 스테이지 ####################################################
FROM arm64v8/python:3.9-slim-bookworm AS builder
ENV DEBIAN_FRONTEND=noninteractive

# ── 공통 저장소(GPG) 설정 ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates apt-transport-https

# Coral EdgeTPU 레포 (libedgetpu·tflite-runtime 제공)
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] "\
         "https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      > /etc/apt/sources.list.d/coral-edgetpu.list

# Raspberry Pi(bookworm) 레포
RUN curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] "\
         "http://archive.raspberrypi.org/debian bookworm main" \
      > /etc/apt/sources.list.d/raspi.list

# ── libcamera 빌드 의존 패키지 ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git meson ninja-build cmake pkg-config \
        pybind11-dev qtbase5-dev \
        libboost-program-options-dev libgnutls28-dev openssl \
        libdrm-dev libudev-dev libexif-dev libyaml-cpp-dev libglib2.0-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libgstreamer-plugins-base1.0-dev libavcodec-dev libavformat-dev libswscale-dev \
        python3-pip

# ── libcamera 소스 빌드 ─────────────────────────────────────────────────────
RUN git clone --depth 1 https://git.libcamera.org/libcamera/libcamera.git /src/libcamera && \
    sed -i '/removeprefix/s/\(.*\)= kebab_case(\(.*\)\.removeprefix(\(.*\)))$/\1 = kebab_case((\2[len(\3):] if \2.startswith(\3) else \2))/' \
        /src/libcamera/utils/codegen/gen-gst-controls.py && \
    meson setup /src/libcamera/build /src/libcamera && \
    ninja -C /src/libcamera/build && ninja -C /src/libcamera/build install && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/libcamera.conf && ldconfig

########## 2) 런타임 스테이지 ##################################################
FROM arm64v8/python:3.9-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# ── 동일 저장소(GPG) 재설정 ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates apt-transport-https && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] "\
         "https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      > /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] "\
         "http://archive.raspberrypi.org/debian bookworm main" \
      > /etc/apt/sources.list.d/raspi.list && \
    apt-get update

# ── 런타임 패키지(※ pycoral 제거) ──────────────────────────────────────────
RUN apt-get install -y --no-install-recommends \
        libedgetpu1-std python3-tflite-runtime \
        libcamera-apps python3-picamera2 \
        libqt5core5a libqt5widgets5 libgstreamer-plugins-base1.0-0 \
        libjpeg62-turbo libpng16-16 libtiff6 libdrm2 libudev1 libglib2.0-0 libexif12 && \
    rm -rf /var/lib/apt/lists/*

# ── 빌드 산출물 복사 ────────────────────────────────────────────────────────
COPY --from=builder /usr/local /usr/local

# ── python ↔ python3 링크 + 경량 pip 패키지 ────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    pip install --no-cache-dir pillow tqdm pyyaml

# ── 애플리케이션 복사 & 기본 실행 ───────────────────────────────────────────
WORKDIR /app
COPY pseudo_processor.py .
ENTRYPOINT ["tail", "-f", "/dev/null"]
# 실제 실행:  ENTRYPOINT ["python", "-u", "pseudo_processor.py"]
