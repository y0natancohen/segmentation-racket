#!/usr/bin/env python3
"""
Complete Pipeline Test for Segmentation-Polygon Integration
Tests the complete pipeline from simulated segmentation to Phaser game.
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

from segmentation.segmentation_polygon_bridge import SegmentationPolygonBridge

class CompletePipelineTest:
    """Test the complete segmentation-to-polygon pipeline"""
    
    def __init__(self):
        self.bridge = SegmentationPolygonBridge(port=8765)
        self.server_task: Optional[asyncio.Task] = None
        self.received_messages: List[Dict[str, Any]] = []
        self.test_duration = 5  # seconds
        self.expected_fps = 10  # Expected FPS from simulated segmentation
        
    async def start_bridge_server(self):
        """Start the bridge server"""
        self.server_task = asyncio.create_task(self.bridge.start_server())
        await asyncio.sleep(1)  # Give server time to start
        logger.info("✅ Bridge server started")
    
    def simulate_segmentation_polygons(self) -> List[np.ndarray]:
        """Simulate segmentation output with various polygon shapes"""
        polygons = []
        
        # Rectangle
        polygons.append(np.array([
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200]
        ], dtype=np.float32))
        
        # Triangle
        polygons.append(np.array([
            [150, 50],
            [100, 150],
            [200, 150]
        ], dtype=np.float32))
        
        # Pentagon
        polygons.append(np.array([
            [150, 50],
            [200, 100],
            [180, 180],
            [120, 180],
            [100, 100]
        ], dtype=np.float32))
        
        # Hexagon
        polygons.append(np.array([
            [150, 50],
            [200, 100],
            [200, 150],
            [150, 200],
            [100, 150],
            [100, 100]
        ], dtype=np.float32))
        
        return polygons
    
    async def simulate_segmentation_loop(self):
        """Simulate the segmentation process sending polygon data"""
        polygons = self.simulate_segmentation_polygons()
        frame_count = 0
        start_time = time.time()
        
        logger.info("🎬 Starting simulated segmentation loop...")
        
        while time.time() - start_time < self.test_duration:
            # Select a polygon (cycle through them)
            polygon = polygons[frame_count % len(polygons)]
            
            # Add some variation to simulate real segmentation
            variation = np.random.normal(0, 5, polygon.shape)
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
                    logger.info(f"📊 Simulated frames: {frame_count}, FPS: {fps:.1f}")
                
            except Exception as e:
                logger.error(f"❌ Error sending polygon: {e}")
            
            # Simulate processing time (similar to real segmentation)
            await asyncio.sleep(0.1)  # 10 FPS
    
    async def test_phaser_connection(self):
        """Test connection from Phaser game perspective"""
        try:
            uri = "ws://localhost:8765"
            logger.info(f"🔌 Connecting to {uri} (Phaser perspective)...")
            
            async with websockets.connect(uri) as websocket:
                logger.info("✅ Connected to bridge WebSocket (Phaser perspective)")
                
                # Collect messages
                start_time = time.time()
                message_count = 0
                
                while time.time() - start_time < self.test_duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        self.received_messages.append(data)
                        message_count += 1
                        
                        # Validate message format
                        if not self.validate_polygon_message(data):
                            logger.error(f"❌ Invalid message format: {data}")
                            return False
                        
                        if message_count % 5 == 0:
                            elapsed = time.time() - start_time
                            fps = message_count / elapsed
                            logger.info(f"📊 Received messages: {message_count}, FPS: {fps:.1f}")
                            
                    except asyncio.TimeoutError:
                        logger.warning("⏰ Timeout waiting for message")
                        break
                
                logger.info(f"✅ Received {message_count} messages from simulated segmentation")
                return True
                
        except Exception as e:
            logger.error(f"❌ Phaser connection test failed: {e}")
            return False
    
    def validate_polygon_message(self, message: Dict[str, Any]) -> bool:
        """Validate that message has correct polygon format"""
        required_fields = ["position", "vertices", "rotation"]
        
        # Check required fields
        for field in required_fields:
            if field not in message:
                logger.error(f"Missing field: {field}")
                return False
        
        # Check position format
        position = message["position"]
        if not isinstance(position, dict) or "x" not in position or "y" not in position:
            logger.error(f"Invalid position format: {position}")
            return False
        
        # Check vertices format
        vertices = message["vertices"]
        if not isinstance(vertices, list) or len(vertices) < 3:
            logger.error(f"Invalid vertices format: {vertices}")
            return False
        
        for vertex in vertices:
            if not isinstance(vertex, dict) or "x" not in vertex or "y" not in vertex:
                logger.error(f"Invalid vertex format: {vertex}")
                return False
        
        # Check rotation format
        if not isinstance(message["rotation"], (int, float)):
            logger.error(f"Invalid rotation format: {message['rotation']}")
            return False
        
        return True
    
    def validate_test_results(self) -> bool:
        """Validate the test results"""
        logger.info("🔍 Validating test results...")
        
        if len(self.received_messages) == 0:
            logger.error("❌ No messages received during test")
            return False
        
        # Check message count and FPS
        actual_fps = len(self.received_messages) / self.test_duration
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Messages received: {len(self.received_messages)}")
        logger.info(f"   Test duration: {self.test_duration:.1f}s")
        logger.info(f"   Actual FPS: {actual_fps:.1f}")
        logger.info(f"   Expected FPS: {self.expected_fps}")
        
        # Check if FPS is reasonable (at least 50% of expected)
        min_acceptable_fps = self.expected_fps * 0.5
        if actual_fps < min_acceptable_fps:
            logger.error(f"❌ FPS too low: {actual_fps:.1f} < {min_acceptable_fps:.1f}")
            return False
        
        # Validate message format consistency
        for i, message in enumerate(self.received_messages):
            if not self.validate_polygon_message(message):
                logger.error(f"❌ Invalid message at index {i}")
                return False
        
        # Check for reasonable polygon data
        valid_polygons = 0
        for message in self.received_messages:
            vertices = message["vertices"]
            if len(vertices) >= 3:  # At least a triangle
                valid_polygons += 1
        
        polygon_ratio = valid_polygons / len(self.received_messages)
        logger.info(f"   Valid polygons: {valid_polygons}/{len(self.received_messages)} ({polygon_ratio:.1%})")
        
        if polygon_ratio < 0.8:  # At least 80% should be valid polygons
            logger.error(f"❌ Too few valid polygons: {polygon_ratio:.1%}")
            return False
        
        logger.info("✅ All validation checks passed!")
        return True
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        logger.info("🧹 Cleanup completed")
    
    async def run_complete_test(self) -> bool:
        """Run the complete pipeline test"""
        logger.info("🧪 Starting Complete Pipeline Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Start bridge server
            logger.info("Step 1: Starting bridge server...")
            await self.start_bridge_server()
            
            # Step 2: Start simulated segmentation and Phaser connection in parallel
            logger.info("Step 2: Starting simulated segmentation and Phaser connection...")
            
            # Run both tasks concurrently
            segmentation_task = asyncio.create_task(self.simulate_segmentation_loop())
            phaser_task = asyncio.create_task(self.test_phaser_connection())
            
            # Wait for both to complete
            segmentation_result, phaser_result = await asyncio.gather(
                segmentation_task, phaser_task, return_exceptions=True
            )
            
            # Check results
            if isinstance(segmentation_result, Exception):
                logger.error(f"❌ Segmentation simulation failed: {segmentation_result}")
                return False
            
            if isinstance(phaser_result, Exception):
                logger.error(f"❌ Phaser connection failed: {phaser_result}")
                return False
            
            if not phaser_result:
                logger.error("❌ Phaser connection test failed")
                return False
            
            # Step 3: Validate results
            logger.info("Step 3: Validating results...")
            return self.validate_test_results()
            
        except Exception as e:
            logger.error(f"❌ Complete pipeline test failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main test function"""
    test = CompletePipelineTest()
    success = await test.run_complete_test()
    
    if success:
        logger.info("🎉 Complete pipeline test PASSED!")
        return 0
    else:
        logger.error("💥 Complete pipeline test FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
