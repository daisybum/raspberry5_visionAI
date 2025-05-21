import os
import time
import logging
from typing import Dict, Any, Optional, List

import redis
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEG] %(levelname)s: %(message)s"
)
logger = logging.getLogger("seg_app")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
ENDPT = os.getenv("TPU_ENDPOINT", "http://tpu_server:8080")
CHANNEL = "img_channel"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def process_image(key: str) -> Optional[Dict[str, Any]]:
    """Process an image using the segmentation endpoint
    
    Args:
        key: Redis key where the image is stored
        
    Returns:
        Segmentation result or None if processing failed
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Processing image {key} (attempt {attempt+1}/{MAX_RETRIES})")
            res = requests.post(
                f"{ENDPT}/segment", 
                json={"redis_key": key}, 
                timeout=15
            )
            res.raise_for_status()
            result = res.json()
            
            # Log detailed results
            mask_shape = result.get("mask_shape", [])
            unique_labels = result.get("unique_labels", [])
            logger.info(f"Segmentation result: mask shape={mask_shape}, labels={unique_labels}")
            
            return result
            
        except RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Max retries reached for {key}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None


def main() -> None:
    """Main function to listen for images and process them"""
    logger.info(f"Starting segmentation client, connecting to {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Using TPU endpoint: {ENDPT}")
    
    # Connect to Redis
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
            p = r.pubsub()
            p.subscribe(CHANNEL)
            logger.info(f"Subscribed to channel: {CHANNEL}")
            break
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    
    # Process messages
    try:
        for msg in p.listen():
            if msg["type"] != "message":
                continue
                
            try:
                key = msg["data"].decode()
                logger.info(f"Received image key: {key}")
                process_image(key)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        try:
            p.unsubscribe()
            r.close()
        except:
            pass


if __name__ == "__main__":
    main()
