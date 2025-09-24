#!/usr/bin/env python3
"""
Test Integration - Simple test to verify the video communication and segmentation integration.
"""

import asyncio
import logging
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_api import init_video_system, get_video_manager
from video_communication import VideoConfig
from segmentation.video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_integration():
    """Test the video communication and segmentation integration."""
    logger.info("Testing video communication and segmentation integration...")
    
    try:
        # Initialize video system
        config = VideoConfig(
            max_samples=30,
            frame_timeout=5.0,
            log_interval=1.0
        )
        init_video_system(config)
        logger.info("Video system initialized")
        
        # Create segmentation processor
        segmentation_args = create_segmentation_args()
        processor = VideoSegmentationProcessor(segmentation_args)
        
        # Initialize processor
        await processor.initialize()
        logger.info("Segmentation processor initialized")
        
        # Test frame processing
        import numpy as np
        
        # Create a test frame
        test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        frame_data = {
            'frame': test_frame,
            'timestamp': 1234567890.0,
            'connection_id': 'test_conn',
            'frame_shape': test_frame.shape
        }
        
        # Process frame
        processor._process_frame(frame_data)
        logger.info("Frame processing test completed")
        
        # Cleanup
        await processor.stop_processing()
        logger.info("Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_integration())
