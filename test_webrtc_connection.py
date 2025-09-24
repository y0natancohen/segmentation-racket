#!/usr/bin/env python3
"""
Test script to simulate a WebRTC connection and trigger segmentation processor.
"""

import asyncio
import aiohttp
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_webrtc_connection():
    """Test WebRTC connection to trigger segmentation processor."""
    
    # Create a mock SDP offer
    mock_offer = {
        "type": "offer",
        "sdp": "v=0\r\no=- 1234567890 1234567890 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=msid-semantic: WMS\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 127.0.0.1\r\na=rtcp:9 IN IP4 127.0.0.1\r\na=ice-ufrag:test\r\na=ice-pwd:test\r\na=fingerprint:sha-256 test\r\na=setup:actpass\r\na=mid:0\r\na=sendonly\r\na=rtcp-mux\r\na=rtpmap:96 VP8/90000\r\n"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            logger.info("🧪 Testing WebRTC connection to trigger segmentation processor...")
            
            # Send offer to server
            async with session.post(
                'http://localhost:8080/offer',
                json=mock_offer,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    answer = await response.json()
                    logger.info(f"✅ Received answer: {answer.get('type', 'unknown')}")
                    logger.info("🎯 This should trigger segmentation processor registration")
                else:
                    logger.error(f"❌ Error: {response.status} - {await response.text()}")
                    
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_webrtc_connection())
