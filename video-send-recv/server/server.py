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
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import object_from_string, object_to_string

from media import mean_intensity
from metrics import MetricsCollector, ConnectionMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global metrics collector
metrics_collector = MetricsCollector()

# Track active connections
active_connections: Dict[str, RTCPeerConnection] = {}
connection_metrics: Dict[str, ConnectionMetrics] = {}
data_channels: Dict[str, any] = {}  # Track data channels by connection ID


class WebRTCServer:
    """WebRTC server for video processing."""
    
    def __init__(self):
        self.app = web.Application()
        self.setup_routes()
    
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
            
            # Create peer connection
            pc = RTCPeerConnection()
            connection_id = f"conn_{int(time.time() * 1000)}"
            active_connections[connection_id] = pc
            connection_metrics[connection_id] = ConnectionMetrics(connection_id)
            
            # Set up event handlers
            self.setup_peer_connection_handlers(pc, connection_id)
            
            # Set remote description
            offer = RTCSessionDescription(sdp=sdp, type=offer_type)
            await pc.setRemoteDescription(offer)
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Return answer
            return web.json_response({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    def setup_peer_connection_handlers(self, pc: RTCPeerConnection, connection_id: str):
        """Set up peer connection event handlers."""
        
        @pc.on("track")
        async def on_track(track):
            """Handle incoming video track."""
            logger.info(f"Received {track.kind} track from {connection_id}")
            
            if track.kind == "video":
                await self.handle_video_track(track, connection_id)
        
        @pc.on("datachannel")
        async def on_datachannel(channel):
            """Handle data channel."""
            logger.info(f"Received data channel: {channel.label} from {connection_id}")
            data_channels[connection_id] = channel  # Store the data channel
            
            @channel.on("open")
            async def on_open():
                logger.info(f"Data channel opened: {channel.label}")
            
            @channel.on("close")
            async def on_close():
                logger.info(f"Data channel closed: {channel.label}")
                if connection_id in data_channels:
                    del data_channels[connection_id]
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            """Handle connection state changes."""
            logger.info(f"Connection state changed to {pc.connectionState} for {connection_id}")
            
            if pc.connectionState in ["closed", "failed", "disconnected"]:
                # Only cleanup if connection still exists
                if connection_id in active_connections:
                    await self.cleanup_connection(connection_id)
    
    async def handle_video_track(self, track, connection_id: str):
        """Process video track and send intensity metrics continuously."""
        logger.info(f"Starting continuous video processing for {connection_id}")
        
        # Keep track of intensity values for averaging
        intensity_values = []
        max_samples = 30  # Keep last 30 frames for averaging
        
        try:
            while True:
                try:
                    # Receive frame with timeout
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    metrics_collector.record_frame_processed()
                    connection_metrics[connection_id].record_frame()
                    
                    # Calculate intensity
                    intensity_255, intensity_norm = mean_intensity(frame)
                    
                    # Add to intensity values for averaging
                    intensity_values.append(intensity_255)
                    if len(intensity_values) > max_samples:
                        intensity_values.pop(0)
                    
                    # Calculate average intensity
                    avg_intensity = sum(intensity_values) / len(intensity_values)
                    avg_intensity_norm = avg_intensity / 255.0
                    
                    # Get data channel for this connection
                    data_channel = data_channels.get(connection_id)
                    
                    # Send metrics if data channel is open
                    if data_channel and data_channel.readyState == "open":
                        metrics_data = {
                            "ts": time.time(),
                            "intensity": intensity_255,
                            "intensity_norm": intensity_norm,
                            "avg_intensity": avg_intensity,
                            "avg_intensity_norm": avg_intensity_norm,
                            "frame_count": len(intensity_values)
                        }
                        
                        try:
                            data_channel.send(json.dumps(metrics_data))
                            metrics_collector.record_message_sent()
                            connection_metrics[connection_id].record_message_sent()
                        except Exception as e:
                            logger.warning(f"Failed to send metrics: {e}")
                    
                    # Log metrics periodically
                    if metrics_collector.should_log():
                        metrics_collector.log_metrics()
                        logger.info(f"Connection {connection_id}: Avg intensity = {avg_intensity:.1f} ({avg_intensity_norm:.3f})")
                
                except asyncio.TimeoutError:
                    logger.warning(f"Frame timeout for connection {connection_id}, continuing...")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing frame for {connection_id}: {e}")
                    raise e
                    continue
                
        except Exception as e:
            logger.error(f"Fatal error in video processing for {connection_id}: {e}")
        finally:
            logger.info(f"Video processing ended for {connection_id}")
            # Don't cleanup connection automatically - let it run forever
    
    async def cleanup_connection(self, connection_id: str):
        """Clean up connection resources."""
        try:
            if connection_id in active_connections:
                pc = active_connections[connection_id]
                await pc.close()
                del active_connections[connection_id]
                logger.info(f"Cleaned up connection {connection_id}")
            
            if connection_id in connection_metrics:
                del connection_metrics[connection_id]
                
            if connection_id in data_channels:
                del data_channels[connection_id]
        except Exception as e:
            logger.warning(f"Error during cleanup of connection {connection_id}: {e}")
    
    async def cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            stale_connections = []
            for conn_id, metrics in connection_metrics.items():
                if metrics.is_stale(timeout=30.0):
                    stale_connections.append(conn_id)
            
            for conn_id in stale_connections:
                logger.info(f"Cleaning up stale connection {conn_id}")
                await self.cleanup_connection(conn_id)


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
