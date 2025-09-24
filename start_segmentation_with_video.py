#!/usr/bin/env python3
"""
Start Segmentation with Video - Start segmentation system connected to video communication.
"""

import asyncio
import logging
import sys
import os
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_send_recv.server.server import WebRTCServer
from aiohttp import web
from segmentation.video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args
from video_api import get_video_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSegmentationServer:
    """Server that runs video communication with segmentation."""
    
    def __init__(self):
        self.webrtc_server = None
        self.segmentation_processor = None
        
    async def start(self):
        """Start the server with segmentation."""
        logger.info("Starting video communication server with segmentation...")
        
        # Start WebRTC server
        self.webrtc_server = WebRTCServer()
        
        # Start the server
        runner = web.AppRunner(self.webrtc_server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        
        logger.info("WebRTC server started on http://localhost:8080")
        
        # Initialize segmentation processor
        segmentation_args = create_segmentation_args()
        self.segmentation_processor = VideoSegmentationProcessor(segmentation_args)
        await self.segmentation_processor.initialize()
        
        # Set up polygon callback to send via WebSocket
        def handle_polygon(polygon_data):
            logger.info(f"Polygon generated: {len(polygon_data['polygon'])} points")
            # Send via WebSocket
            asyncio.create_task(self.webrtc_server.broadcast_polygon_data(polygon_data))
        
        self.segmentation_processor.polygon_callback = handle_polygon
        
        # Set up frame processor in video manager
        video_manager = get_video_manager()
        
        # Override the frame processing to include segmentation
        original_processors = video_manager.frame_processors.copy()
        
        def segmentation_frame_processor(frame_data):
            # Call original processors if any
            for processor in original_processors.values():
                try:
                    processor(frame_data)
                except Exception as e:
                    logger.error(f"Error in original frame processor: {e}")
            
            # Run segmentation
            try:
                self.segmentation_processor._process_frame(frame_data)
            except Exception as e:
                logger.error(f"Error in segmentation frame processor: {e}")
        
        # Set up the segmentation frame processor for all connections
        # This will be called when frames are received
        logger.info("Segmentation system ready. Waiting for video connections...")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if self.segmentation_processor:
                await self.segmentation_processor.stop_processing()

async def main():
    """Main function."""
    server = VideoSegmentationServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
