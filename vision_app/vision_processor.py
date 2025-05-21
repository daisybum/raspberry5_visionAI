from __future__ import annotations

"""Vision Processor

Raspberry Pi + LiteRT 추론 파이프라인
✓ 세그멘테이션 + 분류 모델 동시 실행
✓ Redis 로 결과 전송
✓ libcamera-still 로 주기적 캡처

2025‑05‑21 리팩터링 포인트
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
1. 타입 힌트 오류 수정 (tflite → Interpreter)
2. 양자화 dtype 자동 대응
3. `json` 직렬화로 Redis 호환성 향상
4. XNNPACK delegate 이중 경로 시도
5. `with Image.open` 으로 메모리 안정성
6. `logger.exception` 채택으로 스택 추적 기록
"""

import io
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import redis
from PIL import Image
from tflite_runtime import load_delegate
from tflite_runtime.interpreter import Interpreter

# ─── 로깅 설정 ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VISION] %(levelname)s: %(message)s",
)
logger = logging.getLogger("vision_processor")

# ─── 환경 변수 로드 ────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SEG_MODEL_PATH = Path(os.getenv("SEG_MODEL", "/models/seg_model_int8.tflite"))
CLS_MODEL_PATH = Path(os.getenv("CLS_MODEL", "/models/cls_model_int8.tflite"))
CAPTURE_INTERVAL = int(os.getenv("INTERVAL", "30"))
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
TMP_IMG = Path("/tmp/cap.jpg")
CAP_CMD = [
    "libcamera-still",
    "-n",
    "-o",
    "{dst}",
    "--width",
    "1640",
    "--height",
    "1232",
]

# ─── Delegate 헬퍼 ─────────────────────────────────────────────────────────

def _load_xnnpack_delegate() -> Optional[Any]:
    """XNNPACK delegate 로드 (경로 변종 대응)"""
    for lib in ("libxnnpack.so", "libtensorflowlite_delegate_xnnpack.so"):
        try:
            return load_delegate(lib)
        except Exception:
            continue
    return None

# ─── 모델 로드 ────────────────────────────────────────────────────────────

def load_models() -> Tuple[Interpreter, Interpreter]:
    """세그멘테이션·분류 모델 동시 로드 및 tensors 할당"""
    delegate = _load_xnnpack_delegate()

    def _new_interpreter(model_path: Path) -> Interpreter:
        kwargs = {"model_path": str(model_path), "num_threads": NUM_THREADS}
        if delegate is not None:
            kwargs["experimental_delegates"] = [delegate]
        return Interpreter(**kwargs)

    try:
        logger.info(f"Loading segmentation model → {SEG_MODEL_PATH}")
        seg = _new_interpreter(SEG_MODEL_PATH)
        seg.allocate_tensors()
        seg_shape = seg.get_input_details()[0]["shape"]
        logger.info(f"Seg input size: {seg_shape[2]}×{seg_shape[1]}")

        logger.info(f"Loading classification model → {CLS_MODEL_PATH}")
        cls = _new_interpreter(CLS_MODEL_PATH)
        cls.allocate_tensors()
        cls_shape = cls.get_input_details()[0]["shape"]
        logger.info(f"Cls input size: {cls_shape[2]}×{cls_shape[1]}")

        return seg, cls
    except Exception as exc:
        logger.exception("모델 로드 실패")
        raise exc

# ─── 카메라 캡처 ──────────────────────────────────────────────────────────

def capture_image() -> Optional[bytes]:
    cmd = [part.format(dst=str(TMP_IMG)) for part in CAP_CMD]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Capture failed ({res.returncode}): {res.stderr.strip()}")
        return None
    if not TMP_IMG.exists():
        logger.error("Capture command 성공 → 파일 없음")
        return None
    raw = TMP_IMG.read_bytes()
    TMP_IMG.unlink(missing_ok=True)
    return raw

# ─── 전처리 유틸 ───────────────────────────────────────────────────────────

def _prepare_input(img: Image.Image, target_hw: Tuple[int, int], details: Dict) -> np.ndarray:
    """dtype·양자화 정보에 맞춰 배열 작성"""
    w, h = target_hw
    arr = np.asarray(img.resize((w, h))).astype(np.float32) / 255.0

    dtype = details["dtype"]
    if dtype == np.uint8:
        arr = (arr * 255).astype(np.uint8)
    elif dtype == np.int8:
        scale, zero = details["quantization"]
        arr = ((arr / scale) + zero).astype(np.int8)
    return np.expand_dims(arr, 0)

# ─── 이미지 추론 ───────────────────────────────────────────────────────────

def process_image(
    image_data: bytes,
    seg_interp: Interpreter,
    cls_interp: Interpreter,
) -> Dict[str, Any]:
    ts0 = time.time()
    try:
        with Image.open(io.BytesIO(image_data)) as pil:
            pil = pil.convert("RGB")

            # 세그멘테이션 ———————————————————————————————
            seg_in = seg_interp.get_input_details()[0]
            seg_arr = _prepare_input(pil, (seg_in["shape"][2], seg_in["shape"][1]), seg_in)
            seg_interp.set_tensor(seg_in["index"], seg_arr)
            seg_interp.invoke()
            seg_out = seg_interp.get_output_details()[0]
            mask = seg_interp.get_tensor(seg_out["index"])[0]
            if mask.shape[-1] > 1:
                mask = np.argmax(mask, axis=-1)
            else:
                if mask.dtype != np.float32:
                    mask = mask.astype(np.float32)
                mask = (mask > 0.0).astype(np.uint8)

            # 분류 ———————————————————————————————————
            cls_in = cls_interp.get_input_details()[0]
            cls_arr = _prepare_input(pil, (cls_in["shape"][2], cls_in["shape"][1]), cls_in)
            cls_interp.set_tensor(cls_in["index"], cls_arr)
            cls_interp.invoke()
            cls_out = cls_interp.get_output_details()[0]
            scores = cls_interp.get_tensor(cls_out["index"])[0]
            top_idx = int(scores.argmax())
            top_score = float(scores[top_idx])

        return {
            "segmentation": {
                "mask_shape": mask.shape,
                "unique_labels": np.unique(mask).tolist(),
            },
            "classification": {"id": top_idx, "score": top_score},
            "inference_time_ms": int((time.time() - ts0) * 1000),
        }
    except Exception as exc:
        logger.exception("추론 실패")
        return {
            "error": str(exc),
            "inference_time_ms": int((time.time() - ts0) * 1000),
        }

# ─── Redis 저장 ───────────────────────────────────────────────────────────

def store_result(r: redis.Redis, result: Dict[str, Any]) -> Optional[str]:
    key = f"result:{uuid.uuid4().hex}"
    try:
        r.set(key, json.dumps(result, ensure_ascii=False), ex=3600)
        return key
    except redis.RedisError:
        logger.exception("Redis 저장 실패")
        return None

# ─── 메인 루프 ────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(f"Vision processor 시작 (캡처 주기 {CAPTURE_INTERVAL}s)")

    # 모델 로드
    seg, cls = load_models()

    # Redis 접속
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        logger.info(f"Redis 연결 성공 → {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        logger.exception("Redis 연결 실패·비활성화 모드로 진행")
        r = None

    while True:
        try:
            logger.info("이미지 캡처…")
            data = capture_image()
            if data is None:
                time.sleep(CAPTURE_INTERVAL)
                continue

            logger.info("이미지 추론…")
            result = process_image(data, seg, cls)

            if r is not None:
                key = store_result(r, result)
                if key:
                    logger.info(f"결과 Redis 저장 → {key}")

            logger.info(
                "Seg labels=%s | Cls=(%d, %.3f) | %.1f ms",
                result.get("segmentation", {}).get("unique_labels"),
                result.get("classification", {}).get("id", -1),
                result.get("classification", {}).get("score", 0.0),
                result.get("inference_time_ms", 0),
            )
            time.sleep(CAPTURE_INTERVAL)
        except KeyboardInterrupt:
            logger.info("종료 신호 수신·중단")
            break
        except Exception:
            logger.exception("메인 루프 예외")
            time.sleep(CAPTURE_INTERVAL)


if __name__ == "__main__":
    main()
