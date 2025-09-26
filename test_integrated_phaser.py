#!/usr/bin/env python3
"""
Test script for the integrated Phaser game with video communication and segmentation.

This script tests the integration by running the integrated system and checking connections.
"""

import asyncio
import logging
import sys
import os
import time
import subprocess
import threading
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_send_recv.server.server import WebRTCServer
from aiohttp import web
from segmentation.video_segmentation_integration import run_video_segmentation, create_segmentation_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedPhaserTest:
    """Test the integrated Phaser game system."""
    
    def __init__(self):
        self.webrtc_server = None
        self.segmentation_task = None
        self.running = False
        self.phaser_process = None
    
    async def start_webrtc_server(self, host='localhost', port=8080):
        """Start the WebRTC server."""
        logger.info(f"Starting WebRTC server on {host}:{port}")
        
        self.webrtc_server = WebRTCServer()
        
        # Start the server
        runner = web.AppRunner(self.webrtc_server.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"WebRTC server started on http://{host}:{port}")
        return self.webrtc_server
    
    async def start_segmentation(self):
        """Start the segmentation system."""
        logger.info("Starting segmentation system...")
        
        # Set up polygon data channel with WebSocket server
        from video_segmentation_integration import PolygonDataChannel, get_video_manager, VideoSegmentationProcessor
        
        video_manager = get_video_manager()
        polygon_channel = PolygonDataChannel(video_manager, self.webrtc_server)
        
        # Set up polygon callback
        def handle_polygon(polygon_data):
            polygon_channel.send_polygon(polygon_data)
        
        # Create segmentation arguments
        segmentation_args = create_segmentation_args()
        
        # Initialize processor with callback
        processor = VideoSegmentationProcessor(segmentation_args, handle_polygon)
        await processor.initialize()
        
        # Store processor for later use
        self.segmentation_processor = processor
        self.polygon_channel = polygon_channel
        
        logger.info("Video segmentation processor initialized (waiting for connection)")
        
        # Keep running until stopped
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping video segmentation...")
        finally:
            if hasattr(self, 'segmentation_processor'):
                await self.segmentation_processor.stop_processing()
    
    def start_phaser_game(self):
        """Start the Phaser game in development mode."""
        logger.info("Starting Phaser game...")
        
        phaser_dir = Path(__file__).parent / "phaser-matter-game"
        
        try:
            # Start the development server
            self.phaser_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=phaser_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info("Phaser game started on development server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Phaser game: {e}")
            return False
    
    async def run_test(self, host='localhost', port=8080):
        """Run the integrated test."""
        logger.info("Starting integrated Phaser game test...")
        
        try:
            # Start WebRTC server
            self.webrtc_server = await self.start_webrtc_server(host, port)
            
            # Store reference to integrated system in server
            if hasattr(self.webrtc_server, 'app'):
                self.webrtc_server.app['integrated_system'] = self
                
                # Start segmentation in background
                self.segmentation_task = asyncio.create_task(
                    self.start_segmentation()
                )
                
                # Start Phaser game
                if self.start_phaser_game():
                    logger.info("✅ Integrated system started successfully")
                    logger.info("🌐 Phaser game should be available at http://localhost:5173")
                    logger.info("📡 WebRTC server running on http://localhost:8080")
                    logger.info("🎯 Segmentation system ready for video processing")
                    
                    self.running = True
                    
                    # Keep running
                    while self.running:
                        await asyncio.sleep(1)
                else:
                    logger.error("❌ Failed to start Phaser game")
                    return False
                
        except KeyboardInterrupt:
            logger.info("Shutting down integrated test...")
            self.running = False
            
            if self.segmentation_task:
                self.segmentation_task.cancel()
                try:
                    await self.segmentation_task
                except asyncio.CancelledError:
                    pass
            
            if self.phaser_process:
                self.phaser_process.terminate()
                self.phaser_process.wait()
            
            logger.info("Integrated test shutdown complete")
        
        except Exception as e:
            logger.error(f"Error in integrated test: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

async def main():
    """Main function."""
    logger.info("🚀 Starting Integrated Phaser Game Test")
    
    # Create and run integrated test
    test = IntegratedPhaserTest()
    await test.run_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {e}")
        sys.exit(1)
