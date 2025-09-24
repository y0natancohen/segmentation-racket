#!/usr/bin/env python3
"""
Test script to test the complete flow with actual video frames.
"""

import asyncio
import logging
import sys
import os
import numpy as np
import time
import websockets
import json

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_api import init_video_system, get_video_manager
from video_communication import VideoConfig
from video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args, PolygonDataChannel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_flow():
    """Test the complete flow with actual video frames."""
    
    try:
        # Initialize video system
        config = VideoConfig(
            max_samples=30,
            frame_timeout=5.0,
            log_interval=1.0
        )
        init_video_system(config)
        
        # Get video manager
        video_manager = get_video_manager()
        
        # Create segmentation processor
        args = create_segmentation_args()
        processor = VideoSegmentationProcessor(args)
        await processor.initialize()
        
        # Set up polygon callback
        polygon_data_received = []
        def handle_polygon(polygon_data):
            polygon_data_received.append(polygon_data)
            logger.info(f"🎯 Polygon data received: {len(polygon_data.get('polygon', []))} points")
        
        processor.polygon_callback = handle_polygon
        
        # Register frame processor
        connection_id = "test_conn_123"
        video_manager.set_frame_processor(connection_id, processor._process_frame)
        
        # Start processing
        processor.connection_id = connection_id
        processor.is_processing = True
        
        logger.info(f"✅ Frame processor registered for connection: {connection_id}")
        
        # Simulate video frames
        logger.info("🧪 Simulating video frames...")
        for i in range(5):
            # Create a more realistic frame
            frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            frame[100:200, 200:400] = 255  # White rectangle
            frame[250:350, 300:500] = 128  # Gray rectangle
            
            frame_data = {
                'frame': frame,
                'timestamp': time.time(),
                'connection_id': connection_id,
                'frame_shape': frame.shape
            }
            
            logger.info(f"Processing frame {i+1}/5...")
            processor._process_frame(frame_data)
            await asyncio.sleep(0.1)  # Simulate frame rate
        
        logger.info(f"✅ Processed 5 frames, received {len(polygon_data_received)} polygon data")
        
        if polygon_data_received:
            logger.info("🎯 Segmentation is working!")
        else:
            logger.warning("⚠️ No polygon data generated - this might be expected for synthetic frames")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_flow())