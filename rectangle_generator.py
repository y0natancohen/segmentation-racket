#!/usr/bin/env python3
"""
Rectangle Generator Process
Generates rectangle movement data at 60 FPS and sends via WebSocket to main process.
"""

import asyncio
import websockets
import json
import time
import math
import signal
import sys
from typing import Dict, Any

class RectangleGenerator:
    def __init__(self, host: str = "localhost", port: int = 8765, fps: int = 60):
        self.host = host
        self.port = port
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.running = False
        
        # Rectangle movement parameters
        self.size = 600  # Match the game size
        self.amplitude = self.size * 0.15  # Movement range (0.4 to 0.7 of size)
        self.center_y = self.size * 0.55   # Center position
        self.frequency = 0.5  # 0.5 Hz = 2 second cycle
        self.start_time = time.time()
        
        # WebSocket server
        self.server = None
        self.clients = set()
        
        # Performance monitoring
        self.frame_count = 0
        self.last_stats_time = time.time()
        
    def calculate_rectangle_position(self, current_time: float) -> Dict[str, Any]:
        """Calculate rectangle position using sine wave movement."""
        elapsed_time = current_time - self.start_time
        
        # Sine wave movement: up and down every 2 seconds
        phase = elapsed_time * self.frequency * 2 * math.pi
        y_offset = self.amplitude * math.sin(phase)
        y_position = self.center_y + y_offset
        
        # Calculate velocity for prediction/extrapolation
        velocity_y = self.amplitude * self.frequency * 2 * math.pi * math.cos(phase)
        
        return {
            "timestamp": current_time,
            "position": {
                "x": self.size / 2,  # Fixed X position
                "y": y_position
            },
            "velocity": {
                "x": 0.0,
                "y": velocity_y
            },
            "phase": phase,
            "elapsed_time": elapsed_time
        }
    
    async def register_client(self, websocket, path):
        """Register a new client connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            # Keep connection alive
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_rectangle_data(self):
        """Broadcast rectangle position data to all connected clients."""
        next_frame_time = time.time()
        
        while self.running:
            # Calculate current rectangle position
            current_time = time.time()
            rectangle_data = self.calculate_rectangle_position(current_time)
            
            # Send to all connected clients
            if self.clients:
                message = json.dumps(rectangle_data)
                disconnected_clients = set()
                
                for client in self.clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                self.clients -= disconnected_clients
            
            # Performance monitoring
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_stats_time >= 5.0:  # Print stats every 5 seconds
                actual_fps = self.frame_count / (current_time - self.last_stats_time)
                print(f"Performance: {actual_fps:.1f} FPS, {len(self.clients)} clients")
                self.frame_count = 0
                self.last_stats_time = current_time
            
            # Maintain precise timing for smooth movement
            next_frame_time += self.frame_duration
            current_time = time.time()
            sleep_time = max(0, next_frame_time - current_time)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # If we're behind, skip frames to catch up
                next_frame_time = current_time
    
    async def start_server(self):
        """Start the WebSocket server."""
        print(f"Starting rectangle generator server on {self.host}:{self.port}")
        print(f"Target FPS: {self.fps}")
        print(f"Frame duration: {self.frame_duration:.4f}s ({1000/self.fps:.1f}ms)")
        print(f"Movement parameters: amplitude={self.amplitude}, frequency={self.frequency}Hz")
        
        self.server = await websockets.serve(
            self.register_client,
            self.host,
            self.port,
            ping_interval=1,
            ping_timeout=2
        )
        
        self.running = True
        
        # Start broadcasting in background
        broadcast_task = asyncio.create_task(self.broadcast_rectangle_data())
        
        try:
            # Keep server running
            await self.server.wait_closed()
        except asyncio.CancelledError:
            print("Server shutdown requested")
        finally:
            self.running = False
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
    
    def stop(self):
        """Stop the server gracefully."""
        print("Stopping rectangle generator...")
        self.running = False
        if self.server:
            self.server.close()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start rectangle generator
    generator = RectangleGenerator(fps=60)
    
    try:
        await generator.start_server()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        generator.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
