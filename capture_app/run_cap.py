import os
import time
import uuid
import logging
import subprocess
from pathlib import Path
from typing import Optional

import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CAP] %(levelname)s: %(message)s"
)
logger = logging.getLogger("capture_app")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
INTERVAL = int(os.getenv("INTERVAL", 30))
CHANNEL = "img_channel"
IMAGE_TTL = 60  # seconds
CAP_CMD = ["libcamera-still", "-n", "-o", "{dst}", "--width", "1640", "--height", "1232"]
TMP_IMG = Path("/tmp/cap.jpg")
MAX_REDIS_RETRIES = 3


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


def publish_to_redis(image_data: bytes) -> Optional[str]:
    """Publish image data to Redis
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Redis key if successful, None otherwise
    """
    key = f"img:{uuid.uuid4().hex}"
    
    for attempt in range(MAX_REDIS_RETRIES):
        try:
            # Connect to Redis (create new connection each time to avoid stale connections)
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
            
            # Store image with TTL
            r.setex(key, IMAGE_TTL, image_data)
            
            # Publish notification
            r.publish(CHANNEL, key)
            
            logger.info(f"Published image {key} ({len(image_data)} bytes)")
            return key
            
        except redis.RedisError as e:
            logger.error(f"Redis error (attempt {attempt+1}/{MAX_REDIS_RETRIES}): {e}")
            if attempt < MAX_REDIS_RETRIES - 1:
                time.sleep(1)  # Short delay before retry
        finally:
            try:
                r.close()
            except:
                pass
    
    return None


def main() -> None:
    """Main capture loop"""
    logger.info(f"Starting capture service with {INTERVAL}s interval")
    logger.info(f"Images will be published to Redis {REDIS_HOST}:{REDIS_PORT} on channel {CHANNEL}")
    
    while True:
        try:
            # Capture image
            image_data = capture_image()
            if not image_data:
                logger.warning("Capture failed, will retry after interval")
                time.sleep(INTERVAL)
                continue
                
            # Publish to Redis
            if publish_to_redis(image_data):
                # Success - wait for next interval
                time.sleep(INTERVAL)
            else:
                # Redis publishing failed
                logger.error("Failed to publish to Redis, will retry after interval")
                time.sleep(INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Shutting down capture service")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
