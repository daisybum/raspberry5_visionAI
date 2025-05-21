from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import os
import io
import logging
import redis

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from PIL import Image
from filelock import FileLock

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment, classify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TPU] %(levelname)s: %(message)s"
)
logger = logging.getLogger("tpu_server")

# Configuration
SEG_PATH = os.getenv("SEG_MODEL", "/models/seg_edgetpu.tflite")
CLS_PATH = os.getenv("CLS_MODEL", "/models/cls_edgetpu.tflite")
LOCK_PATH = Path("/tmp/edgetpu.lock")          # 컨테이너 내부 전역 락
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ─── Interpreter preload ──────────────────────────────────
try:
    logger.info(f"Loading segmentation model from {SEG_PATH}")
    seg_interp = edgetpu.make_interpreter(str(SEG_PATH), device="usb")
    seg_interp.allocate_tensors()
    seg_w, seg_h = common.input_size(seg_interp)
    logger.info(f"Segmentation model loaded, input size: {seg_w}x{seg_h}")
    
    logger.info(f"Loading classification model from {CLS_PATH}")
    cls_interp = edgetpu.make_interpreter(str(CLS_PATH), device="usb")
    cls_interp.allocate_tensors()
    cls_w, cls_h = common.input_size(cls_interp)
    logger.info(f"Classification model loaded, input size: {cls_w}x{cls_h}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# ─── FastAPI 라우터 ────────────────────────────────────────
app = FastAPI(
    title="Edge-TPU Gateway", 
    version="1.0",
    description="API for accessing Edge TPU models for segmentation and classification"
)

# Redis connection
try:
    r = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        decode_responses=False,
        socket_connect_timeout=5
    )
    r.ping()  # Test connection
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    # We'll continue and retry connections later

class ImgReq(BaseModel):
    """Request model for image processing endpoints"""
    redis_key: str          # Key set by seg_app or cls_app


def _load_image_from_redis(key: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load image from Redis and preprocess for model input
    
    Args:
        key: Redis key where image is stored
        target_size: Target size (width, height) for resizing
        
    Returns:
        Preprocessed image as numpy array
        
    Raises:
        HTTPException: If image not found or processing fails
    """
    try:
        # Try to get image from Redis
        raw = r.get(key)
        if raw is None:
            logger.warning(f"Image key not found: {key}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Image key expired or not found"
            )
            
        # Process image
        pil = Image.open(io.BytesIO(raw)).convert("RGB").resize(target_size)
        return np.asarray(pil).astype(np.uint8)
    except redis.RedisError as e:
        logger.error(f"Redis error while loading image {key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis connection error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing image {key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/segment", response_model=Dict[str, Any])
async def segment_image(req: ImgReq) -> Dict[str, Any]:
    """Process image with segmentation model
    
    Args:
        req: Request containing Redis key for image
        
    Returns:
        Dictionary with segmentation results
    """
    logger.info(f"Processing segmentation request for key: {req.redis_key}")
    try:
        # Load image from Redis
        np_img = _load_image_from_redis(req.redis_key, (seg_w, seg_h))
        
        # Process with Edge TPU
        with FileLock(LOCK_PATH):
            logger.debug("Acquired TPU lock for segmentation")
            common.set_input(seg_interp, np_img)
            seg_interp.invoke()
            mask = segment.get_output(seg_interp)
            logger.debug("Released TPU lock for segmentation")
        
        # Post-process result
        if mask.ndim == 3:
            mask = np.argmax(mask, axis=-1).astype(np.uint8)
        
        # Clean up Redis
        try:
            r.delete(req.redis_key)
            logger.debug(f"Deleted Redis key: {req.redis_key}")
        except redis.RedisError as e:
            logger.warning(f"Failed to delete Redis key {req.redis_key}: {e}")
        
        # Return results
        result = {
            "mask_shape": list(mask.shape),
            "unique_labels": np.unique(mask).tolist()
        }
        logger.info(f"Segmentation complete: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/classify", response_model=Dict[str, Any])
async def classify_image(req: ImgReq) -> Dict[str, Any]:
    """Process image with classification model
    
    Args:
        req: Request containing Redis key for image
        
    Returns:
        Dictionary with classification results
    """
    logger.info(f"Processing classification request for key: {req.redis_key}")
    try:
        # Load image from Redis
        np_img = _load_image_from_redis(req.redis_key, (cls_w, cls_h))
        
        # Process with Edge TPU
        with FileLock(LOCK_PATH):
            logger.debug("Acquired TPU lock for classification")
            common.set_input(cls_interp, np_img)
            cls_interp.invoke()
            pred = classify.get_classes(cls_interp, top_k=1)[0]
            logger.debug("Released TPU lock for classification")
        
        # Convert NumPy types to Python native types
        result = {
            "id": int(pred.id),
            "score": float(pred.score)
        }
        
        # Clean up Redis
        try:
            r.delete(req.redis_key)
            logger.debug(f"Deleted Redis key: {req.redis_key}")
        except redis.RedisError as e:
            logger.warning(f"Failed to delete Redis key {req.redis_key}: {e}")
        
        logger.info(f"Classification complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )
