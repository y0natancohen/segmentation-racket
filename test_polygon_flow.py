#!/usr/bin/env python3
"""
Test Polygon Flow - Test the complete polygon data flow from segmentation to WebSocket.
"""

import asyncio
import json
import logging
import sys
import os
import time
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_send_recv.server.server import WebRTCServer
from aiohttp import web
from segmentation.video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PolygonFlowTest:
    """Test the complete polygon data flow."""
    
    def __init__(self):
        self.webrtc_server = None
        self.polygon_data_received = []
        
    async def start_server(self):
        """Start the WebRTC server."""
        logger.info("Starting WebRTC server...")
        self.webrtc_server = WebRTCServer()
        
        # Start the server
        runner = web.AppRunner(self.webrtc_server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        
        logger.info("WebRTC server started on http://localhost:8080")
    
    async def test_polygon_broadcast(self):
        """Test broadcasting polygon data."""
        logger.info("Testing polygon data broadcast...")
        
        # Create test polygon data
        test_polygon_data = {
            'connection_id': 'test_conn',
            'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
            'timestamp': time.time(),
            'frame_shape': [360, 640]
        }
        
        # Broadcast the test data
        await self.webrtc_server.broadcast_polygon_data(test_polygon_data)
        logger.info("Test polygon data broadcasted")
        
        # Wait a bit for the broadcast
        await asyncio.sleep(1)
    
    async def test_segmentation_processor(self):
        """Test the segmentation processor."""
        logger.info("Testing segmentation processor...")
        
        # Create segmentation processor
        segmentation_args = create_segmentation_args()
        processor = VideoSegmentationProcessor(segmentation_args)
        
        # Initialize processor
        await processor.initialize()
        logger.info("Segmentation processor initialized")
        
        # Create test frame data with a simple shape that should generate a polygon
        test_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Add a simple rectangle in the center
        test_frame[100:200, 200:400] = [255, 255, 255]  # White rectangle
        frame_data = {
            'frame': test_frame,
            'timestamp': time.time(),
            'connection_id': 'test_conn',
            'frame_shape': test_frame.shape
        }
        
        # Set up polygon callback
        def handle_polygon(polygon_data):
            logger.info(f"Polygon received: {polygon_data}")
            self.polygon_data_received.append(polygon_data)
            
            # Broadcast to WebSocket clients
            asyncio.create_task(self.webrtc_server.broadcast_polygon_data(polygon_data))
        
        processor.polygon_callback = handle_polygon
        
        # Process frame
        processor._process_frame(frame_data)
        logger.info("Frame processed")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Cleanup
        await processor.stop_processing()
        
        return len(self.polygon_data_received) > 0
    
    async def run_test(self):
        """Run the complete test."""
        logger.info("Starting polygon flow test...")
        
        try:
            # Start server
            await self.start_server()
            
            # Test polygon broadcast
            await self.test_polygon_broadcast()
            
            # Test segmentation processor
            segmentation_worked = await self.test_segmentation_processor()
            
            if segmentation_worked:
                logger.info("✅ Polygon flow test PASSED")
            else:
                logger.error("❌ Polygon flow test FAILED - No polygon data generated")
            
            # Keep server running for a bit
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            logger.info("Test completed")

async def main():
    """Main test function."""
    test = PolygonFlowTest()
    await test.run_test()

if __name__ == "__main__":
    asyncio.run(main())
