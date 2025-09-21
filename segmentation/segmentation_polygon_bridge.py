#!/usr/bin/env python3
"""
Segmentation Polygon Bridge
Connects the segmentation process to the polygon generator system.
Takes polygon data from segmentation and sends it via WebSocket to the Phaser game.
"""

import asyncio
import websockets
import json
import time
import numpy as np
from typing import Optional, Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationPolygonBridge:
    """Bridge between segmentation process and polygon generator system"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.running = False
        
        # Performance monitoring
        self.message_count = 0
        self.last_stats_time = time.time()
    
    async def register_client(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            # Keep connection alive
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_polygon_data(self, polygon_vertices: np.ndarray, 
                              position: Optional[Dict[str, float]] = None,
                              rotation: float = 0.0,
                              frame_size: tuple = (640, 480)):
        """
        Send polygon data from segmentation to connected clients.
        
        Args:
            polygon_vertices: Nx2 numpy array of (x,y) vertices from segmentation
            position: Optional position dict with 'x' and 'y' keys
            rotation: Rotation angle in radians
            frame_size: (width, height) of the frame for coordinate scaling
        """
        if not self.clients or polygon_vertices is None:
            return
        
        # Scale coordinates from camera frame to game dimensions
        frame_width, frame_height = frame_size
        scale_x = 800 / frame_width  # Game width / frame width
        scale_y = 450 / frame_height  # Game height / frame height
        
        # Scale polygon vertices
        scaled_vertices = polygon_vertices.copy()
        scaled_vertices[:, 0] = polygon_vertices[:, 0] * scale_x
        scaled_vertices[:, 1] = polygon_vertices[:, 1] * scale_y
        
        # Convert numpy array to list of dicts
        vertices = []
        for vertex in scaled_vertices:
            vertices.append({"x": float(vertex[0]), "y": float(vertex[1])})
        
        # Use provided position or calculate center from scaled polygon
        if position is None:
            # Calculate center of scaled polygon
            center_x = float(np.mean(scaled_vertices[:, 0]))
            center_y = float(np.mean(scaled_vertices[:, 1]))
            position = {"x": center_x, "y": center_y}
        else:
            # Scale the provided position as well
            position = {
                "x": position["x"] * scale_x,
                "y": position["y"] * scale_y
            }
        
        # Create message in the same format as polygon_generator
        message = {
            "position": position,
            "vertices": vertices,
            "rotation": rotation
        }
        
        # Send to all connected clients
        message_json = json.dumps(message)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message_json)
                self.message_count += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.warning(f"Error sending message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
        
        # Log performance stats every 5 seconds
        current_time = time.time()
        if current_time - self.last_stats_time >= 5.0:
            fps = self.message_count / (current_time - self.last_stats_time)
            logger.info(f"Sent {self.message_count} messages, FPS: {fps:.1f}, Clients: {len(self.clients)}")
            self.message_count = 0
            self.last_stats_time = current_time
    
    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting segmentation polygon bridge on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.register_client,
            self.host,
            self.port
        )
        
        self.running = True
        logger.info("Segmentation polygon bridge server started")
        
        try:
            # Keep server running
            await self.server.wait_closed()
        except asyncio.CancelledError:
            pass
    
    def stop(self):
        """Stop the server gracefully."""
        logger.info("Stopping segmentation polygon bridge...")
        self.running = False
        if self.server:
            self.server.close()

# Global bridge instance for use by segmentation process
_bridge_instance: Optional[SegmentationPolygonBridge] = None

def get_bridge() -> Optional[SegmentationPolygonBridge]:
    """Get the global bridge instance."""
    return _bridge_instance

def set_bridge(bridge: SegmentationPolygonBridge):
    """Set the global bridge instance."""
    global _bridge_instance
    _bridge_instance = bridge

async def send_segmentation_polygon(polygon_vertices: np.ndarray, 
                                  position: Optional[Dict[str, float]] = None,
                                  rotation: float = 0.0,
                                  frame_size: tuple = (640, 480)):
    """
    Convenience function to send polygon data from segmentation process.
    This can be called from the segmentation loop.
    """
    bridge = get_bridge()
    if bridge and bridge.running:
        await bridge.send_polygon_data(polygon_vertices, position, rotation, frame_size)
