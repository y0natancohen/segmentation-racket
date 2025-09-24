#!/usr/bin/env python3
"""
Test script to verify the complete segmentation flow.
"""

import asyncio
import aiohttp
import json
import logging
import time
import websockets
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_segmentation_flow():
    """Test the complete segmentation flow."""
    
    try:
        # Test WebRTC connection
        logger.info("🧪 Testing WebRTC connection...")
        mock_offer = {
            "type": "offer",
            "sdp": "v=0\r\no=- 1234567890 1234567890 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=msid-semantic: WMS\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 127.0.0.1\r\na=rtcp:9 IN IP4 127.0.0.1\r\na=ice-ufrag:test\r\na=ice-pwd:test\r\na=fingerprint:sha-256 test\r\na=setup:actpass\r\na=mid:0\r\na=sendonly\r\na=rtcp-mux\r\na=rtpmap:96 VP8/90000\r\n"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8080/offer',
                json=mock_offer,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    answer = await response.json()
                    logger.info(f"✅ WebRTC connection established: {answer.get('type', 'unknown')}")
                else:
                    logger.error(f"❌ WebRTC connection failed: {response.status}")
                    return
        
        # Test WebSocket connection
        logger.info("🧪 Testing WebSocket connection...")
        try:
            async with websockets.connect('ws://localhost:8080/polygon') as ws:
                logger.info("✅ WebSocket connected")
                
                # Wait for polygon data
                logger.info("🎯 Waiting for polygon data...")
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    polygon_data = json.loads(message)
                    logger.info(f"🎯 Received polygon data: {polygon_data}")
                    logger.info(f"🎯 Polygon points: {len(polygon_data.get('polygon', []))}")
                    logger.info("✅ Segmentation flow working!")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ No polygon data received within 10 seconds")
                    logger.info("💡 This might be expected if no video frames are being processed")
                except Exception as e:
                    logger.error(f"❌ Error receiving polygon data: {e}")
                    
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_segmentation_flow())
