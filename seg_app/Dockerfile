FROM arm64v8/python:3.8-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# 최소 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# Python 라이브러리
RUN pip install --no-cache-dir \
        redis~=5.0 \
        requests~=2.32 \
        numpy~=1.24

WORKDIR /app
COPY run_seg.py .

ENTRYPOINT ["python", "-u", "run_seg.py"]
