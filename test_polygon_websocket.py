#!/usr/bin/env python3
"""
Test Polygon WebSocket - Test sending polygon data via WebSocket.
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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_polygon_websocket():
    """Test WebSocket polygon data flow."""
    logger.info("Starting WebSocket polygon test...")
    
    try:
        # Start WebRTC server
        webrtc_server = WebRTCServer()
        
        # Start the server
        runner = web.AppRunner(webrtc_server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        
        logger.info("WebRTC server started on http://localhost:8080")
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Send test polygon data
        test_polygon_data = {
            'connection_id': 'test_conn',
            'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
            'timestamp': time.time(),
            'frame_shape': [360, 640]
        }
        
        logger.info("Sending test polygon data...")
        await webrtc_server.broadcast_polygon_data(test_polygon_data)
        
        # Send more test data
        for i in range(5):
            test_polygon_data = {
                'connection_id': 'test_conn',
                'polygon': [
                    [50 + i*20, 50 + i*20], 
                    [150 + i*20, 50 + i*20], 
                    [150 + i*20, 150 + i*20], 
                    [50 + i*20, 150 + i*20]
                ],
                'timestamp': time.time(),
                'frame_shape': [360, 640]
            }
            
            logger.info(f"Sending test polygon {i+1}...")
            await webrtc_server.broadcast_polygon_data(test_polygon_data)
            await asyncio.sleep(1)
        
        logger.info("✅ WebSocket polygon test completed")
        
        # Keep server running
        await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_polygon_websocket())
