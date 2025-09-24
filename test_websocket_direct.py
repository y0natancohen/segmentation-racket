#!/usr/bin/env python3
"""
Test WebSocket Direct - Test WebSocket connection directly.
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

async def test_websocket_direct():
    """Test WebSocket connection directly."""
    logger.info("Testing WebSocket connection...")
    
    try:
        # Connect to WebSocket
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
                
                # Wait for response
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        logger.info(f"Received: {msg.data}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                
                logger.info("✅ WebSocket test completed")
                
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket_direct())
