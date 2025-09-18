#!/usr/bin/env python3
"""
End-to-End Test for Segmentation-Polygon Integration
Tests the complete pipeline from segmentation to Phaser game polygon display.
"""

import asyncio
import websockets
import json
import time
import numpy as np
import subprocess
import sys
import os
import signal
import threading
from typing import Optional, Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationPolygonIntegrationTest:
    """Test the complete segmentation-to-polygon pipeline"""
    
    def __init__(self):
        self.segmentation_process: Optional[subprocess.Popen] = None
        self.phaser_process: Optional[subprocess.Popen] = None
        self.received_messages: List[Dict[str, Any]] = []
        self.test_duration = 10  # seconds
        self.expected_fps = 15  # Expected FPS from segmentation
        
    async def test_websocket_connection(self, port: int = 8765) -> bool:
        """Test WebSocket connection to segmentation bridge"""
        try:
            uri = f"ws://localhost:{port}"
            async with websockets.connect(uri) as websocket:
                logger.info(f"✅ Connected to segmentation bridge at {uri}")
                
                # Wait for messages
                start_time = time.time()
                message_count = 0
                
                while time.time() - start_time < 5:  # Wait up to 5 seconds
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        self.received_messages.append(data)
                        message_count += 1
                        
                        # Validate message format
                        if not self.validate_polygon_message(data):
                            logger.error(f"❌ Invalid message format: {data}")
                            return False
                            
                    except asyncio.TimeoutError:
                        logger.warning("⏰ Timeout waiting for message")
                        break
                
                if message_count > 0:
                    logger.info(f"✅ Received {message_count} valid polygon messages")
                    return True
                else:
                    logger.error("❌ No messages received")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
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
        if not isinstance(vertices, list) or len(vertices) == 0:
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
    
    def start_segmentation_process(self) -> bool:
        """Start the segmentation process with polygon bridge enabled"""
        try:
            cmd = [
                sys.executable, "segmentation.py",
                "--polygon_bridge",
                "--polygon_bridge_port", "8765",
                "--web_display",
                "--web_port", "8080",
                "--cam", "0",
                "--width", "640",
                "--height", "480",
                "--headless"  # Run without GUI
            ]
            
            logger.info(f"🚀 Starting segmentation process: {' '.join(cmd)}")
            self.segmentation_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the process to start
            time.sleep(3)
            
            if self.segmentation_process.poll() is None:
                logger.info("✅ Segmentation process started successfully")
                return True
            else:
                stdout, stderr = self.segmentation_process.communicate()
                logger.error(f"❌ Segmentation process failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start segmentation process: {e}")
            return False
    
    def start_phaser_game(self) -> bool:
        """Start the Phaser game process"""
        try:
            cmd = ["npm", "run", "dev"]
            cwd = "/home/jonathan/segment_project/phaser-matter-game"
            
            logger.info(f"🚀 Starting Phaser game: {' '.join(cmd)}")
            self.phaser_process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the game to start
            time.sleep(5)
            
            if self.phaser_process.poll() is None:
                logger.info("✅ Phaser game started successfully")
                return True
            else:
                stdout, stderr = self.phaser_process.communicate()
                logger.error(f"❌ Phaser game failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start Phaser game: {e}")
            return False
    
    def cleanup_processes(self):
        """Clean up all started processes"""
        logger.info("🧹 Cleaning up processes...")
        
        if self.segmentation_process:
            try:
                self.segmentation_process.terminate()
                self.segmentation_process.wait(timeout=5)
                logger.info("✅ Segmentation process terminated")
            except subprocess.TimeoutExpired:
                self.segmentation_process.kill()
                logger.info("🔪 Segmentation process killed")
            except Exception as e:
                logger.warning(f"⚠️ Error terminating segmentation process: {e}")
        
        if self.phaser_process:
            try:
                self.phaser_process.terminate()
                self.phaser_process.wait(timeout=5)
                logger.info("✅ Phaser process terminated")
            except subprocess.TimeoutExpired:
                self.phaser_process.kill()
                logger.info("🔪 Phaser process killed")
            except Exception as e:
                logger.warning(f"⚠️ Error terminating Phaser process: {e}")
        
        # Additional cleanup
        try:
            subprocess.run(["pkill", "-f", "segmentation.py"], check=False)
            subprocess.run(["pkill", "-f", "npm run dev"], check=False)
            subprocess.run(["pkill", "-f", "vite"], check=False)
        except Exception as e:
            logger.warning(f"⚠️ Error in additional cleanup: {e}")
    
    async def run_integration_test(self) -> bool:
        """Run the complete integration test"""
        logger.info("🧪 Starting Segmentation-Polygon Integration Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Start segmentation process
            logger.info("Step 1: Starting segmentation process...")
            if not self.start_segmentation_process():
                return False
            
            # Step 2: Test WebSocket connection
            logger.info("Step 2: Testing WebSocket connection...")
            if not await self.test_websocket_connection():
                return False
            
            # Step 3: Start Phaser game
            logger.info("Step 3: Starting Phaser game...")
            if not self.start_phaser_game():
                return False
            
            # Step 4: Monitor message flow
            logger.info("Step 4: Monitoring message flow...")
            await self.monitor_message_flow()
            
            # Step 5: Validate results
            logger.info("Step 5: Validating results...")
            return self.validate_test_results()
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
        finally:
            self.cleanup_processes()
    
    async def monitor_message_flow(self):
        """Monitor the message flow for a period of time"""
        start_time = time.time()
        message_count = 0
        
        try:
            uri = "ws://localhost:8765"
            async with websockets.connect(uri) as websocket:
                logger.info("📡 Monitoring message flow...")
                
                while time.time() - start_time < self.test_duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        self.received_messages.append(data)
                        message_count += 1
                        
                        if message_count % 10 == 0:
                            elapsed = time.time() - start_time
                            current_fps = message_count / elapsed
                            logger.info(f"📊 Messages: {message_count}, FPS: {current_fps:.1f}")
                            
                    except asyncio.TimeoutError:
                        logger.warning("⏰ Timeout waiting for message")
                        break
                        
        except Exception as e:
            logger.error(f"❌ Error monitoring message flow: {e}")
    
    def validate_test_results(self) -> bool:
        """Validate the test results"""
        logger.info("🔍 Validating test results...")
        
        if len(self.received_messages) == 0:
            logger.error("❌ No messages received during test")
            return False
        
        # Check message count and FPS
        elapsed_time = self.test_duration
        actual_fps = len(self.received_messages) / elapsed_time
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Messages received: {len(self.received_messages)}")
        logger.info(f"   Test duration: {elapsed_time:.1f}s")
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
        
        if polygon_ratio < 0.5:  # At least 50% should be valid polygons
            logger.error(f"❌ Too few valid polygons: {polygon_ratio:.1%}")
            return False
        
        logger.info("✅ All validation checks passed!")
        return True

async def main():
    """Main test function"""
    test = SegmentationPolygonIntegrationTest()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("🛑 Received interrupt signal, cleaning up...")
        test.cleanup_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the test
    success = await test.run_integration_test()
    
    if success:
        logger.info("🎉 Integration test PASSED!")
        return 0
    else:
        logger.error("💥 Integration test FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
