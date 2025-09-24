#!/usr/bin/env python3
"""
Send Test Polygon - Send test polygon data to the WebSocket server.
"""

import asyncio
import json
import logging
import sys
import os
import time
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def send_test_polygon():
    """Send test polygon data to the WebSocket server."""
    logger.info("Connecting to WebSocket server...")
    
    try:
        # Connect to WebSocket
        uri = "ws://localhost:8080/polygon"
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")
            
            # Send test polygon data
            for i in range(10):
                test_polygon_data = {
                    'connection_id': 'test_conn',
                    'polygon': [
                        [100 + i*10, 100 + i*10], 
                        [200 + i*10, 100 + i*10], 
                        [200 + i*10, 200 + i*10], 
                        [100 + i*10, 200 + i*10]
                    ],
                    'timestamp': time.time(),
                    'frame_shape': [360, 640]
                }
                
                logger.info(f"Sending test polygon {i+1}...")
                await websocket.send(json.dumps(test_polygon_data))
                await asyncio.sleep(1)
            
            logger.info("✅ Test polygon data sent successfully")
            
    except Exception as e:
        logger.error(f"Failed to send test polygon data: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_polygon())
