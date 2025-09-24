#!/usr/bin/env python3
"""
Send Polygon Manual - Manually send polygon data to test the WebSocket flow.
"""

import asyncio
import json
import logging
import sys
import os
import time
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def send_polygon_manual():
    """Manually send polygon data to test WebSocket flow."""
    logger.info("Manually sending polygon data...")
    
    try:
        # Connect to WebSocket and send data
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect('http://localhost:8080/polygon') as ws:
                logger.info("Connected to WebSocket server")
                
                # Send test polygon data
                test_polygon_data = {
                    'connection_id': 'test_conn',
                    'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
                    'timestamp': time.time(),
                    'frame_shape': [360, 640]
                }
                
                logger.info("Sending test polygon data...")
                await ws.send_str(json.dumps(test_polygon_data))
                
                # Send a few more polygons
                for i in range(3):
                    test_polygon_data = {
                        'connection_id': 'test_conn',
                        'polygon': [
                            [50 + i*30, 50 + i*30], 
                            [150 + i*30, 50 + i*30], 
                            [150 + i*30, 150 + i*30], 
                            [50 + i*30, 150 + i*30]
                        ],
                        'timestamp': time.time(),
                        'frame_shape': [360, 640]
                    }
                    
                    logger.info(f"Sending polygon {i+1}...")
                    await ws.send_str(json.dumps(test_polygon_data))
                    await asyncio.sleep(1)
                
                logger.info("✅ Manual polygon data sent")
                
    except Exception as e:
        logger.error(f"Failed to send polygon data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(send_polygon_manual())
