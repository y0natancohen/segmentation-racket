#!/usr/bin/env python3
"""
Test script to directly test the segmentation processor with synthetic frames.
"""

import asyncio
import logging
import sys
import os
import numpy as np
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_segmentation_direct():
    """Test segmentation processor directly with synthetic frames."""
    
    try:
        # Create segmentation args
        args = create_segmentation_args()
        logger.info(f"Created segmentation args: {args}")
        
        # Create processor
        processor = VideoSegmentationProcessor(args)
        await processor.initialize()
        logger.info("✅ Segmentation processor initialized")
        
        # Create synthetic frame (person-like shape)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Draw a simple person-like shape
        # Head (circle)
        cv2.circle(frame, (320, 100), 30, (255, 255, 255), -1)
        
        # Body (rectangle)
        cv2.rectangle(frame, (300, 130), (340, 250), (255, 255, 255), -1)
        
        # Arms
        cv2.rectangle(frame, (280, 150), (300, 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (340, 150), (360, 200), (255, 255, 255), -1)
        
        # Legs
        cv2.rectangle(frame, (310, 250), (330, 350), (255, 255, 255), -1)
        cv2.rectangle(frame, (330, 250), (350, 350), (255, 255, 255), -1)
        
        logger.info(f"Created synthetic frame: {frame.shape}")
        
        # Test frame processing
        frame_data = {
            'frame': frame,
            'timestamp': time.time(),
            'connection_id': 'test_conn_123',
            'frame_shape': frame.shape
        }
        
        logger.info("🧪 Testing frame processing...")
        processor._process_frame(frame_data)
        
        logger.info("✅ Segmentation test completed")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import cv2
    asyncio.run(test_segmentation_direct())