#!/usr/bin/env python3
"""
Test script to verify frame processor registration and calling.
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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_frame_processor():
    """Test that frame processor is registered and called."""
    
    # Initialize video system
    config = VideoConfig(
        max_samples=30,
        frame_timeout=5.0,
        log_interval=1.0
    )
    init_video_system(config)
    
    # Get video manager
    video_manager = get_video_manager()
    
    # Test frame processor registration
    def test_processor(frame_data):
        logger.info(f"🎯 Frame processor called with frame_data keys: {frame_data.keys()}")
        logger.info(f"🎯 Frame shape: {frame_data.get('frame_shape', 'unknown')}")
        logger.info(f"🎯 Connection ID: {frame_data.get('connection_id', 'unknown')}")
    
    # Register processor
    test_connection_id = "test_conn_123"
    video_manager.set_frame_processor(test_connection_id, test_processor)
    
    # Simulate frame data
    import numpy as np
    test_frame_data = {
        'frame': np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8),
        'timestamp': 1234567890.0,
        'connection_id': test_connection_id,
        'frame_shape': (360, 640, 3)
    }
    
    # Call processor directly
    logger.info("🧪 Testing frame processor directly...")
    test_processor(test_frame_data)
    
    # Test removal
    video_manager.remove_frame_processor(test_connection_id)
    logger.info("✅ Frame processor test completed")

if __name__ == "__main__":
    asyncio.run(test_frame_processor())
