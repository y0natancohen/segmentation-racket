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
from aiohttp.web import WebSocketResponse

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
        self.websocket_clients = set()
        
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
        self.app.router.add_get('/polygon', self.handle_polygon_websocket)
    
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
            
            # Start segmentation processor for this connection if integrated system is available
            if hasattr(self, 'app') and 'integrated_system' in self.app:
                integrated_system = self.app['integrated_system']
                if hasattr(integrated_system, 'start_segmentation_for_connection'):
                    await integrated_system.start_segmentation_for_connection(connection_id)
            
            # Return answer
            return web.json_response(answer)
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def handle_polygon_websocket(self, request):
        """Handle WebSocket connection for polygon data."""
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_clients.add(ws)
        logger.info(f"Polygon WebSocket client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle incoming messages if needed
                    pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websocket_clients.discard(ws)
            logger.info(f"Polygon WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
        
        return ws
    
    async def broadcast_polygon_data(self, polygon_data: dict):
        """Broadcast polygon data to all connected WebSocket clients."""
        logger.info(f"📡 Broadcasting polygon data to {len(self.websocket_clients)} clients")
        logger.info(f"📡 Polygon data: {polygon_data}")
        
        if not self.websocket_clients:
            logger.warning("No WebSocket clients connected")
            return
        
        message = json.dumps(polygon_data)
        disconnected_clients = set()
        
        for ws in self.websocket_clients:
            try:
                await ws.send_str(message)
                logger.info(f"✅ Sent polygon data to client")
            except Exception as e:
                logger.error(f"Error sending polygon data to client: {e}")
                disconnected_clients.add(ws)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
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
