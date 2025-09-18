#!/usr/bin/env python3
"""
Demo script for Segmentation-Polygon Integration
Demonstrates how to use the integrated system.
"""

import asyncio
import websockets
import json
import time
import numpy as np
import sys
import os
import subprocess
import threading
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation_polygon_bridge import SegmentationPolygonBridge

class SegmentationPolygonDemo:
    """Demo the segmentation-polygon integration"""
    
    def __init__(self):
        self.bridge = SegmentationPolygonBridge(port=8765)
        self.server_task: Optional[asyncio.Task] = None
        
    async def start_bridge_server(self):
        """Start the bridge server"""
        self.server_task = asyncio.create_task(self.bridge.start_server())
        await asyncio.sleep(1)  # Give server time to start
        logger.info("✅ Bridge server started on port 8765")
    
    def create_demo_polygons(self) -> List[np.ndarray]:
        """Create demo polygons that simulate segmentation output"""
        polygons = []
        
        # Person silhouette (rectangle-like)
        polygons.append(np.array([
            [200, 100],  # Head
            [180, 200],  # Shoulder left
            [220, 200],  # Shoulder right
            [200, 100]   # Back to head
        ], dtype=np.float32))
        
        # Hand gesture (triangle-like)
        polygons.append(np.array([
            [150, 150],
            [200, 100],
            [250, 150]
        ], dtype=np.float32))
        
        # Object (pentagon-like)
        polygons.append(np.array([
            [300, 100],
            [350, 120],
            [340, 180],
            [310, 200],
            [280, 150]
        ], dtype=np.float32))
        
        return polygons
    
    async def simulate_segmentation_output(self):
        """Simulate segmentation output with realistic polygon data"""
        polygons = self.create_demo_polygons()
        frame_count = 0
        start_time = time.time()
        
        logger.info("🎬 Simulating segmentation output...")
        logger.info("   This simulates what the segmentation process would send")
        logger.info("   to the Phaser game via WebSocket.")
        
        while time.time() - start_time < 10:  # Run for 10 seconds
            # Select a polygon (cycle through them)
            polygon = polygons[frame_count % len(polygons)]
            
            # Add some realistic variation to simulate real segmentation
            variation = np.random.normal(0, 3, polygon.shape)
            varied_polygon = polygon + variation
            
            # Send polygon data
            try:
                await self.bridge.send_polygon_data(
                    varied_polygon,
                    frame_size=(640, 480)
                )
                frame_count += 1
                
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"📊 Sent {frame_count} polygons, FPS: {fps:.1f}")
                
            except Exception as e:
                logger.error(f"❌ Error sending polygon: {e}")
            
            # Simulate processing time (similar to real segmentation)
            await asyncio.sleep(0.1)  # 10 FPS
    
    async def monitor_phaser_connection(self):
        """Monitor what the Phaser game would receive"""
        try:
            uri = "ws://localhost:8765"
            logger.info(f"🔌 Connecting to {uri} (Phaser game perspective)...")
            
            async with websockets.connect(uri) as websocket:
                logger.info("✅ Connected to bridge WebSocket (Phaser game perspective)")
                logger.info("   This simulates what the Phaser game receives")
                
                message_count = 0
                start_time = time.time()
                
                while time.time() - start_time < 10:  # Run for 10 seconds
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        message_count += 1
                        
                        # Display message details
                        if message_count % 5 == 0:
                            logger.info(f"📥 Message {message_count}:")
                            logger.info(f"   Position: ({data['position']['x']:.1f}, {data['position']['y']:.1f})")
                            logger.info(f"   Vertices: {len(data['vertices'])} points")
                            logger.info(f"   Rotation: {data['rotation']:.2f} rad")
                            
                            # Show first few vertices
                            vertices = data['vertices'][:3]
                            logger.info(f"   Sample vertices: {vertices}")
                            
                    except asyncio.TimeoutError:
                        logger.warning("⏰ Timeout waiting for message")
                        break
                
                elapsed = time.time() - start_time
                fps = message_count / elapsed
                logger.info(f"📊 Phaser received {message_count} messages, FPS: {fps:.1f}")
                
        except Exception as e:
            logger.error(f"❌ Phaser connection monitoring failed: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        logger.info("🧹 Cleanup completed")
    
    async def run_demo(self):
        """Run the complete demo"""
        logger.info("🎭 Segmentation-Polygon Integration Demo")
        logger.info("=" * 60)
        logger.info("This demo shows how segmentation data flows to the Phaser game:")
        logger.info("1. Segmentation process detects objects and creates polygons")
        logger.info("2. Polygon bridge sends data via WebSocket")
        logger.info("3. Phaser game receives and displays the polygons")
        logger.info("")
        
        try:
            # Step 1: Start bridge server
            logger.info("Step 1: Starting polygon bridge server...")
            await self.start_bridge_server()
            
            # Step 2: Run segmentation simulation and Phaser monitoring in parallel
            logger.info("Step 2: Running segmentation simulation and Phaser monitoring...")
            logger.info("")
            
            # Run both tasks concurrently
            segmentation_task = asyncio.create_task(self.simulate_segmentation_output())
            phaser_task = asyncio.create_task(self.monitor_phaser_connection())
            
            # Wait for both to complete
            await asyncio.gather(segmentation_task, phaser_task, return_exceptions=True)
            
            logger.info("")
            logger.info("✅ Demo completed successfully!")
            logger.info("")
            logger.info("To use this in practice:")
            logger.info("1. Run: python3 segmentation.py --polygon_bridge")
            logger.info("2. Start Phaser game: cd phaser-matter-game && npm run dev")
            logger.info("3. The game will receive real-time polygon data from segmentation!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
        finally:
            await self.cleanup()

async def main():
    """Main demo function"""
    demo = SegmentationPolygonDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
