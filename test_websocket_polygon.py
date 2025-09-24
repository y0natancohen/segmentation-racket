#!/usr/bin/env python3
"""
Test WebSocket Polygon - Test the WebSocket polygon data flow without segmentation.
"""

import asyncio
import json
import logging
import sys
import os
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))

from video_send_recv.server.server import WebRTCServer
from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_polygon():
    """Test WebSocket polygon data flow."""
    logger.info("Testing WebSocket polygon data flow...")
    
    try:
        # Start WebRTC server
        webrtc_server = WebRTCServer()
        
        # Start the server
        runner = web.AppRunner(webrtc_server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        
        logger.info("WebRTC server started on http://localhost:8080")
        
        # Wait a bit for server to start
        await asyncio.sleep(1)
        
        # Send test polygon data
        test_polygon_data = {
            'connection_id': 'test_conn',
            'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
            'timestamp': time.time(),
            'frame_shape': [360, 640]
        }
        
        logger.info("Sending test polygon data...")
        await webrtc_server.broadcast_polygon_data(test_polygon_data)
        
        # Send a few more test polygons
        for i in range(5):
            test_polygon_data = {
                'connection_id': 'test_conn',
                'polygon': [[50 + i*10, 50 + i*10], [150 + i*10, 50 + i*10], [150 + i*10, 150 + i*10], [50 + i*10, 150 + i*10]],
                'timestamp': time.time(),
                'frame_shape': [360, 640]
            }
            
            logger.info(f"Sending test polygon {i+1}...")
            await webrtc_server.broadcast_polygon_data(test_polygon_data)
            await asyncio.sleep(0.5)
        
        logger.info("✅ WebSocket polygon test completed")
        
        # Keep server running for a bit
        await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket_polygon())
