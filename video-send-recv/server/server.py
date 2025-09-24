"""
WebRTC server for video intensity analysis.
"""
import asyncio
import json
import logging
import time
from typing import Dict, Optional, Tuple

import aiohttp
from aiohttp import web

from video_api import init_video_system, connect_video, get_video_manager
from video_communication import VideoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebRTCServer:
    """WebRTC server for video processing."""
    
    def __init__(self):
        self.app = web.Application()
        self.setup_routes()
        
        # Initialize video system
        config = VideoConfig(
            max_samples=30,
            frame_timeout=5.0,
            log_interval=1.0
        )
        init_video_system(config)
    
    def setup_routes(self):
        """Set up HTTP routes."""
        # Add CORS middleware
        @web.middleware
        async def cors_middleware(request, handler):
            # Handle preflight requests
            if request.method == 'OPTIONS':
                response = web.Response()
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                response.headers['Access-Control-Max-Age'] = '86400'
                return response
            
            # Handle actual requests
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        self.app.middlewares.append(cors_middleware)
        
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_post('/offer', self.handle_offer)
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({"ok": True})
    
    
    async def handle_offer(self, request):
        """Handle WebRTC offer and create answer."""
        try:
            data = await request.json()
            sdp = data.get('sdp')
            offer_type = data.get('type')
            
            if not sdp or offer_type != 'offer':
                return web.json_response(
                    {"error": "Invalid offer"}, 
                    status=400
                )
            
            # Generate connection ID
            connection_id = f"conn_{int(time.time() * 1000)}"
            
            # Use the video API to handle the connection
            answer = await connect_video(connection_id, sdp, offer_type)
            
            # Return answer
            return web.json_response(answer)
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Get video manager to check for stale connections
            video_manager = get_video_manager()
            all_connections = video_manager.get_all_connections()
            
            stale_connections = []
            for conn_id, stats in all_connections.items():
                if stats and stats.get('last_activity', 0) < time.time() - 30:
                    stale_connections.append(conn_id)
            
            for conn_id in stale_connections:
                logger.info(f"Cleaning up stale connection {conn_id}")
                await video_manager.cleanup_connection(conn_id)


async def main():
    """Main server function."""
    server = WebRTCServer()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(server.cleanup_stale_connections())
    
    try:
        # Start web server
        runner = web.AppRunner(server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        
        logger.info("WebRTC server started on http://localhost:8080")
        logger.info("Health check: http://localhost:8080/health")
        
        # Keep running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        cleanup_task.cancel()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
