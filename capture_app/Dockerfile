# capture_app/Dockerfile
FROM arm64v8/python:3.8-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# ── 1) APT 저장소 및 libcamera-apps ─────────────────────────────
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates curl gnupg; \
    \
    # Raspberry Pi 저장소 (libcamera 바이너리 제공)
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian bullseye main" \
        > /etc/apt/sources.list.d/raspi.list; \
    \
    apt-get update; \
    apt-get install -y --no-install-recommends libcamera-apps; \
    rm -rf /var/lib/apt/lists/*

# ── 2) Python 라이브러리 ────────────────────────────────────────
RUN pip install --no-cache-dir \
        redis~=5.0 \
        numpy~=1.24

WORKDIR /app
COPY run_cap.py .

ENTRYPOINT ["python", "-u", "run_cap.py"]
