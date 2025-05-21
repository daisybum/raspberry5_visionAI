FROM arm64v8/python:3.8-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# ── (1) APT 저장소 등록 & 필수 패키지 ─────────────────────────────
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates curl gnupg build-essential; \
    \
    ## ── Coral EdgeTPU 저장소 GPG 키 추가
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list; \
    \
    ## ── Raspberry Pi 저장소 GPG 키 추가
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian bullseye main" \
        > /etc/apt/sources.list.d/raspi.list; \
    \
    apt-get update; \
    \
    ## ── Edge-TPU 런타임 & TFLite
    apt-get install -y --no-install-recommends \
        libedgetpu1-std python3-pycoral python3-tflite-runtime; \
    \
    ## ── 캐시 정리
    rm -rf /var/lib/apt/lists/*

# ── (2) Python 패키지 ────────────────────────────────────────────
RUN pip install --no-cache-dir \
        fastapi~=0.111 uvicorn[standard]~=0.30 \
        numpy pillow redis requests filelock &&\
    python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

WORKDIR /app
COPY main.py .
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
