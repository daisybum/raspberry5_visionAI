import os
import time
import uuid
import logging
import subprocess
import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import redis
import numpy as np
from PIL import Image
# LiteRT를 사용하기 위한 임포트
from tflite_runtime import Interpreter
from tflite_runtime import load_delegate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VISION] %(levelname)s: %(message)s"
)
logger = logging.getLogger("vision_processor")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SEG_MODEL_PATH = os.getenv("SEG_MODEL", "/models/seg_model.tflite")
CLS_MODEL_PATH = os.getenv("CLS_MODEL", "/models/cls_model.tflite")
CAPTURE_INTERVAL = int(os.getenv("INTERVAL", "30"))
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
TMP_IMG = Path("/tmp/cap.jpg")
CAP_CMD = ["libcamera-still", "-n", "-o", "{dst}", "--width", "1640", "--height", "1232"]
MAX_REDIS_RETRIES = 3

# ─── Model Loading ──────────────────────────────────────
def load_models() -> Tuple[Interpreter, Interpreter]:
    """Load TFLite models
    
    Returns:
        Tuple of (segmentation_interpreter, classification_interpreter)
    """
    try:
        # Load segmentation model
        logger.info(f"Loading segmentation model from {SEG_MODEL_PATH}")
        # LiteRT를 사용하여 세그멘테이션 모델 로드 (XNNPACK 델리게이트 사용)
        try:
            # 먼저 XNNPACK 델리게이트 사용 시도
            xnnpack_delegate = load_delegate('libxnnpack.so')
            seg_interp = Interpreter(
                model_path=str(SEG_MODEL_PATH),
                num_threads=NUM_THREADS,
                experimental_delegates=[xnnpack_delegate]
            )
        except Exception as e:
            logger.warning(f"XNNPACK 델리게이트 로드 실패, 기본 인터프리터 사용: {e}")
            seg_interp = Interpreter(
                model_path=str(SEG_MODEL_PATH),
                num_threads=NUM_THREADS
            )
        seg_interp.allocate_tensors()
        
        # Get input and output details
        seg_input_details = seg_interp.get_input_details()
        seg_output_details = seg_interp.get_output_details()
        seg_input_shape = seg_input_details[0]['shape']
        seg_h, seg_w = seg_input_shape[1], seg_input_shape[2]
        logger.info(f"Segmentation model loaded, input size: {seg_w}x{seg_h}")
        
        # Load classification model
        logger.info(f"Loading classification model from {CLS_MODEL_PATH}")
        # LiteRT를 사용하여 분류 모델 로드 (XNNPACK 델리게이트 사용)
        try:
            # 먼저 XNNPACK 델리게이트 사용 시도
            xnnpack_delegate = load_delegate('libxnnpack.so')
            cls_interp = Interpreter(
                model_path=str(CLS_MODEL_PATH),
                num_threads=NUM_THREADS,
                experimental_delegates=[xnnpack_delegate]
            )
        except Exception as e:
            logger.warning(f"XNNPACK 델리게이트 로드 실패, 기본 인터프리터 사용: {e}")
            cls_interp = Interpreter(
                model_path=str(CLS_MODEL_PATH),
                num_threads=NUM_THREADS
            )
        cls_interp.allocate_tensors()
        
        # Get input and output details
        cls_input_details = cls_interp.get_input_details()
        cls_output_details = cls_interp.get_output_details()
        cls_input_shape = cls_input_details[0]['shape']
        cls_h, cls_w = cls_input_shape[1], cls_input_shape[2]
        logger.info(f"Classification model loaded, input size: {cls_w}x{cls_h}")
        
        return seg_interp, cls_interp
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# ─── Image Capture ──────────────────────────────────────
def capture_image() -> Optional[bytes]:
    """Capture an image using libcamera-still
    
    Returns:
        Raw image bytes if successful, None otherwise
    """
    try:
        # Use subprocess instead of os.system for better error handling
        cmd = [part.format(dst=TMP_IMG) if "{dst}" in part else part for part in CAP_CMD]
        logger.debug(f"Running capture command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Capture failed with code {result.returncode}: {result.stderr}")
            return None
            
        if not TMP_IMG.exists():
            logger.error("Capture command succeeded but no image file was created")
            return None
            
        # Read image and clean up
        raw = TMP_IMG.read_bytes()
        TMP_IMG.unlink(missing_ok=True)
        logger.debug(f"Captured image: {len(raw)} bytes")
        return raw
        
    except Exception as e:
        logger.error(f"Error during image capture: {e}")
        return None

# ─── Image Processing ────────────────────────────────────
def process_image(
    image_data: bytes, 
    seg_interp: tflite.Interpreter, 
    cls_interp: tflite.Interpreter
) -> Dict[str, Any]:
    """Process image with both segmentation and classification models
    
    Args:
        image_data: Raw image bytes
        seg_interp: Segmentation model interpreter
        cls_interp: Classification model interpreter
        
    Returns:
        Dictionary with combined results
    """
    start_time = time.time()
    
    try:
        # Load and preprocess image
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get input details
        seg_input_details = seg_interp.get_input_details()
        seg_output_details = seg_interp.get_output_details()
        cls_input_details = cls_interp.get_input_details()
        cls_output_details = cls_interp.get_output_details()
        
        # Get input shapes
        seg_input_shape = seg_input_details[0]['shape']
        cls_input_shape = cls_input_details[0]['shape']
        seg_h, seg_w = seg_input_shape[1], seg_input_shape[2]
        cls_h, cls_w = cls_input_shape[1], cls_input_shape[2]
        
        # Segmentation inference
        seg_img = pil_img.resize((seg_w, seg_h))
        seg_array = np.asarray(seg_img).astype(np.float32) / 255.0
        seg_array = np.expand_dims(seg_array, axis=0)
        
        seg_interp.set_tensor(seg_input_details[0]['index'], seg_array)
        seg_interp.invoke()
        mask = seg_interp.get_tensor(seg_output_details[0]['index'])
        
        # Post-process segmentation result
        if mask.ndim == 4:  # [batch, height, width, classes]
            mask = mask[0]  # Remove batch dimension
        
        if mask.shape[-1] > 1:  # Multiple classes
            mask = np.argmax(mask, axis=-1).astype(np.uint8)
        else:  # Binary mask
            mask = (mask > 0.5).astype(np.uint8)
        
        # Classification inference
        cls_img = pil_img.resize((cls_w, cls_h))
        cls_array = np.asarray(cls_img).astype(np.float32) / 255.0
        cls_array = np.expand_dims(cls_array, axis=0)
        
        cls_interp.set_tensor(cls_input_details[0]['index'], cls_array)
        cls_interp.invoke()
        output_data = cls_interp.get_tensor(cls_output_details[0]['index'])
        
        # Post-process classification result
        scores = output_data[0]
        top_class_idx = np.argmax(scores)
        top_score = float(scores[top_class_idx])
        
        # Return combined results
        result = {
            "segmentation": {
                "mask_shape": list(mask.shape),
                "unique_labels": np.unique(mask).tolist(),
            },
            "classification": {
                "id": int(top_class_idx),
                "score": top_score,
            },
            "inference_time_ms": int((time.time() - start_time) * 1000)
        }
        
        logger.info(f"Inference complete in {result['inference_time_ms']}ms")
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {
            "error": str(e),
            "inference_time_ms": int((time.time() - start_time) * 1000)
        }

# ─── Redis Operations ─────────────────────────────────────
def store_result_in_redis(result: Dict[str, Any], r: redis.Redis) -> str:
    """Store inference result in Redis
    
    Args:
        result: Inference result dictionary
        r: Redis connection
        
    Returns:
        Redis key where result is stored
    """
    key = f"result:{uuid.uuid4().hex}"
    try:
        r.set(key, str(result), ex=3600)  # Store for 1 hour
        logger.debug(f"Stored result in Redis: {key}")
        return key
    except redis.RedisError as e:
        logger.error(f"Failed to store result in Redis: {e}")
        return ""

# ─── Main Loop ────────────────────────────────────────────
def main() -> None:
    """Main processing loop"""
    logger.info(f"Starting vision processor with {CAPTURE_INTERVAL}s interval")
    
    # Load models
    try:
        seg_interp, cls_interp = load_models()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return
    
    # Connect to Redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()  # Test connection
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        r = None
    
    # Main processing loop
    while True:
        try:
            # 1. Capture image
            logger.info("Capturing image...")
            image_data = capture_image()
            if not image_data:
                logger.warning("Capture failed, will retry after interval")
                time.sleep(CAPTURE_INTERVAL)
                continue
            
            # 2. Process image
            logger.info("Processing image...")
            result = process_image(image_data, seg_interp, cls_interp)
            
            # 3. Store result in Redis if available
            if r:
                try:
                    result_key = store_result_in_redis(result, r)
                    if result_key:
                        logger.info(f"Result stored in Redis: {result_key}")
                except Exception as e:
                    logger.error(f"Redis error: {e}")
            
            # 4. Log detailed results
            seg_result = result.get("segmentation", {})
            cls_result = result.get("classification", {})
            
            logger.info("Processing results:")
            logger.info(f"  Segmentation: mask shape={seg_result.get('mask_shape', [])}, labels={seg_result.get('unique_labels', [])}")
            logger.info(f"  Classification: class={cls_result.get('id', -1)}, score={cls_result.get('score', 0.0):.4f}")
            logger.info(f"  Inference time: {result.get('inference_time_ms', 0)}ms")
            
            # 5. Wait for next interval
            logger.info(f"Waiting {CAPTURE_INTERVAL}s until next capture...")
            time.sleep(CAPTURE_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Shutting down vision processor")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(CAPTURE_INTERVAL)


if __name__ == "__main__":
    main()
