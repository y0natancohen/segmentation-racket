#!/usr/bin/env python3
"""
Integrated System Runner - Runs the video communication system with segmentation.

This script starts both the WebRTC server and the segmentation system integrated together.
"""

import asyncio
import logging
import sys
import os
import argparse
import threading
import time

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
logger.setLevel(logging.DEBUG)

class IntegratedSystem:
    """Manages the integrated video communication and segmentation system."""
    
    def __init__(self, segmentation_args):
        self.segmentation_args = segmentation_args
        self.webrtc_server = None
        self.segmentation_task = None
        self.running = False
    
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
    
    async def start_segmentation(self, connection_id: str = None):
        """Start the segmentation system."""
        logger.info("Starting segmentation system...")
        
        # Set up polygon data channel with WebSocket server
        from video_segmentation_integration import PolygonDataChannel, get_video_manager, VideoSegmentationProcessor
        
        video_manager = get_video_manager()
        polygon_channel = PolygonDataChannel(video_manager, self.webrtc_server)
        
        # Set up polygon callback
        def handle_polygon(polygon_data):
            polygon_channel.send_polygon(polygon_data)
        
        # Initialize processor with callback
        processor = VideoSegmentationProcessor(self.segmentation_args, handle_polygon)
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
    
    async def start_segmentation_for_connection(self, connection_id: str):
        """Start segmentation processing for a specific connection."""
        if hasattr(self, 'segmentation_processor') and self.segmentation_processor:
            logger.info(f"🎯 Starting segmentation processing for connection: {connection_id}")
            await self.segmentation_processor.start_processing(connection_id)
            self.polygon_channel.set_connection(connection_id)
            logger.info(f"🎯 Segmentation processor registered for connection: {connection_id}")
        else:
            logger.warning("⚠️ Segmentation processor not initialized")
    
    async def run(self, host='localhost', port=8080, connection_id: str = None):
        """Run the integrated system."""
        logger.info("Starting integrated video communication and segmentation system...")
        
        try:
            # Start WebRTC server
            self.webrtc_server = await self.start_webrtc_server(host, port)
            
            # Store reference to integrated system in server
            if hasattr(self.webrtc_server, 'app'):
                self.webrtc_server.app['integrated_system'] = self
                
                # Start segmentation in background
                self.segmentation_task = asyncio.create_task(
                    self.start_segmentation(connection_id)
                )
                
                self.running = True
                logger.info("Integrated system started successfully")
                
                # Keep running
                while self.running:
                    await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down integrated system...")
            self.running = False
            
            if self.segmentation_task:
                self.segmentation_task.cancel()
                try:
                    await self.segmentation_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Integrated system shutdown complete")
        
        except Exception as e:
            logger.error(f"Error in integrated system: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run integrated video communication and segmentation system')
    
    # Server options
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    # Segmentation options
    parser.add_argument('--model', default='rvm_mobilenetv3.pth', help='Segmentation model')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--polygon_threshold', type=float, default=0.5, help='Polygon threshold')
    parser.add_argument('--polygon_min_area', type=int, default=2000, help='Minimum polygon area')
    parser.add_argument('--polygon_epsilon', type=float, default=0.015, help='Polygon epsilon')
    
    # Connection options
    parser.add_argument('--connection_id', help='Specific connection ID to process')
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_args()
    
    # Create segmentation arguments
    segmentation_args = create_segmentation_args()
    segmentation_args.model = args.model
    segmentation_args.polygon_threshold = args.polygon_threshold
    segmentation_args.polygon_min_area = args.polygon_min_area
    segmentation_args.polygon_epsilon = args.polygon_epsilon
    
    # Set device
    if args.device == 'auto':
        import torch
        segmentation_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        segmentation_args.device = args.device
    
    # Create and run integrated system
    system = IntegratedSystem(segmentation_args)
    await system.run(args.host, args.port, args.connection_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
