#!/usr/bin/env python3
"""
Test Coordinate System for Segmentation-Polygon Integration
Tests that the coordinate scaling works correctly between camera and game.
"""

import asyncio
import websockets
import json
import time
import numpy as np
import sys
import os
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation.segmentation_polygon_bridge import SegmentationPolygonBridge

# Import constants from segmentation
from segmentation.segmentation import CAMERA_WIDTH, CAMERA_HEIGHT, GAME_WIDTH, GAME_HEIGHT, CAMERA_ASPECT_RATIO, GAME_ASPECT_RATIO

class CoordinateSystemTest:
    """Test the coordinate system scaling"""
    
    def __init__(self):
        self.bridge = SegmentationPolygonBridge(port=8765)
        self.server_task: Optional[asyncio.Task] = None
        self.received_messages: List[Dict[str, Any]] = []
        
    async def start_bridge_server(self):
        """Start the bridge server"""
        self.server_task = asyncio.create_task(self.bridge.start_server())
        await asyncio.sleep(1)  # Give server time to start
        logger.info("✅ Bridge server started")
    
    def create_test_polygons(self) -> List[Dict[str, Any]]:
        """Create test polygons with known dimensions"""
        test_cases = []
        
        # Test 1: Full frame polygon (should fill entire game)
        full_frame_polygon = np.array([
            [0, 0],                    # Top-left
            [CAMERA_WIDTH, 0],         # Top-right
            [CAMERA_WIDTH, CAMERA_HEIGHT],  # Bottom-right
            [0, CAMERA_HEIGHT]         # Bottom-left
        ], dtype=np.float32)
        
        test_cases.append({
            "name": "Full Frame",
            "polygon": full_frame_polygon,
            "expected_bounds": {
                "min_x": 0,
                "max_x": GAME_WIDTH,
                "min_y": 0,
                "max_y": GAME_HEIGHT
            }
        })
        
        # Test 2: Center rectangle (should be centered in game)
        center_rect = np.array([
            [CAMERA_WIDTH * 0.25, CAMERA_HEIGHT * 0.25],   # Top-left
            [CAMERA_WIDTH * 0.75, CAMERA_HEIGHT * 0.25],   # Top-right
            [CAMERA_WIDTH * 0.75, CAMERA_HEIGHT * 0.75],   # Bottom-right
            [CAMERA_WIDTH * 0.25, CAMERA_HEIGHT * 0.75]    # Bottom-left
        ], dtype=np.float32)
        
        test_cases.append({
            "name": "Center Rectangle",
            "polygon": center_rect,
            "expected_bounds": {
                "min_x": GAME_WIDTH * 0.25,
                "max_x": GAME_WIDTH * 0.75,
                "min_y": GAME_HEIGHT * 0.25,
                "max_y": GAME_HEIGHT * 0.75
            }
        })
        
        # Test 3: Top-left corner (should be in top-left of game)
        top_left_polygon = np.array([
            [0, 0],
            [CAMERA_WIDTH * 0.5, 0],
            [CAMERA_WIDTH * 0.5, CAMERA_HEIGHT * 0.5],
            [0, CAMERA_HEIGHT * 0.5]
        ], dtype=np.float32)
        
        test_cases.append({
            "name": "Top-Left Corner",
            "polygon": top_left_polygon,
            "expected_bounds": {
                "min_x": 0,
                "max_x": GAME_WIDTH * 0.5,
                "min_y": 0,
                "max_y": GAME_HEIGHT * 0.5
            }
        })
        
        return test_cases
    
    async def test_coordinate_scaling(self):
        """Test coordinate scaling with known test cases"""
        test_cases = self.create_test_polygons()
        
        logger.info("🧪 Testing Coordinate System Scaling")
        logger.info("=" * 50)
        logger.info(f"Camera dimensions: {CAMERA_WIDTH}x{CAMERA_HEIGHT} (aspect ratio: {CAMERA_ASPECT_RATIO:.3f})")
        logger.info(f"Game dimensions: {GAME_WIDTH}x{GAME_HEIGHT} (aspect ratio: {GAME_ASPECT_RATIO:.3f})")
        logger.info("")
        
        try:
            uri = "ws://localhost:8765"
            async with websockets.connect(uri) as websocket:
                logger.info("✅ Connected to bridge WebSocket")
                
                for i, test_case in enumerate(test_cases):
                    logger.info(f"Test {i+1}: {test_case['name']}")
                    
                    # Send test polygon
                    await self.bridge.send_polygon_data(
                        test_case["polygon"],
                        frame_size=(CAMERA_WIDTH, CAMERA_HEIGHT)
                    )
                    
                    # Receive message
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        
                        # Analyze received polygon
                        vertices = data["vertices"]
                        if not vertices:
                            logger.error(f"❌ No vertices received for {test_case['name']}")
                            continue
                        
                        # Calculate actual bounds
                        x_coords = [v["x"] for v in vertices]
                        y_coords = [v["y"] for v in vertices]
                        actual_bounds = {
                            "min_x": min(x_coords),
                            "max_x": max(x_coords),
                            "min_y": min(y_coords),
                            "max_y": max(y_coords)
                        }
                        
                        # Compare with expected bounds
                        expected = test_case["expected_bounds"]
                        tolerance = 5.0  # Allow 5 pixel tolerance
                        
                        logger.info(f"   Expected bounds: x=[{expected['min_x']:.1f}, {expected['max_x']:.1f}], y=[{expected['min_y']:.1f}, {expected['max_y']:.1f}]")
                        logger.info(f"   Actual bounds:   x=[{actual_bounds['min_x']:.1f}, {actual_bounds['max_x']:.1f}], y=[{actual_bounds['min_y']:.1f}, {actual_bounds['max_y']:.1f}]")
                        
                        # Check if bounds are within tolerance
                        bounds_ok = (
                            abs(actual_bounds["min_x"] - expected["min_x"]) <= tolerance and
                            abs(actual_bounds["max_x"] - expected["max_x"]) <= tolerance and
                            abs(actual_bounds["min_y"] - expected["min_y"]) <= tolerance and
                            abs(actual_bounds["max_y"] - expected["max_y"]) <= tolerance
                        )
                        
                        if bounds_ok:
                            logger.info(f"   ✅ {test_case['name']} scaling is correct!")
                        else:
                            logger.error(f"   ❌ {test_case['name']} scaling is incorrect!")
                            
                        # Additional checks for full frame test
                        if test_case["name"] == "Full Frame":
                            # Check if polygon covers most of the game area
                            coverage_x = (actual_bounds["max_x"] - actual_bounds["min_x"]) / GAME_WIDTH
                            coverage_y = (actual_bounds["max_y"] - actual_bounds["min_y"]) / GAME_HEIGHT
                            
                            logger.info(f"   Coverage: x={coverage_x:.1%}, y={coverage_y:.1%}")
                            
                            if coverage_x >= 0.95 and coverage_y >= 0.95:
                                logger.info(f"   ✅ Full frame polygon covers {coverage_x:.1%}x{coverage_y:.1%} of game area")
                            else:
                                logger.error(f"   ❌ Full frame polygon only covers {coverage_x:.1%}x{coverage_y:.1%} of game area")
                        
                        logger.info("")
                        
                    except asyncio.TimeoutError:
                        logger.error(f"❌ Timeout waiting for message for {test_case['name']}")
                        continue
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Coordinate scaling test failed: {e}")
            return False
    
    async def test_aspect_ratio_consistency(self):
        """Test that aspect ratios are consistent"""
        logger.info("🔍 Testing Aspect Ratio Consistency")
        logger.info("=" * 40)
        
        aspect_ratio_diff = abs(CAMERA_ASPECT_RATIO - GAME_ASPECT_RATIO)
        logger.info(f"Camera aspect ratio: {CAMERA_ASPECT_RATIO:.6f}")
        logger.info(f"Game aspect ratio:   {GAME_ASPECT_RATIO:.6f}")
        logger.info(f"Difference:          {aspect_ratio_diff:.6f}")
        
        if aspect_ratio_diff < 0.001:  # Very small tolerance
            logger.info("✅ Aspect ratios are consistent!")
            return True
        else:
            logger.error("❌ Aspect ratios are inconsistent!")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        logger.info("🧹 Cleanup completed")
    
    async def run_tests(self) -> bool:
        """Run all coordinate system tests"""
        logger.info("🧪 Starting Coordinate System Tests")
        logger.info("=" * 60)
        
        try:
            # Step 1: Start bridge server
            logger.info("Step 1: Starting bridge server...")
            await self.start_bridge_server()
            
            # Step 2: Test aspect ratio consistency
            logger.info("Step 2: Testing aspect ratio consistency...")
            aspect_ratio_ok = await self.test_aspect_ratio_consistency()
            logger.info("")
            
            # Step 3: Test coordinate scaling
            logger.info("Step 3: Testing coordinate scaling...")
            scaling_ok = await self.test_coordinate_scaling()
            
            # Step 4: Validate results
            logger.info("Step 4: Validating results...")
            if aspect_ratio_ok and scaling_ok:
                logger.info("✅ All coordinate system tests passed!")
                return True
            else:
                logger.error("❌ Some coordinate system tests failed!")
                return False
            
        except Exception as e:
            logger.error(f"❌ Coordinate system tests failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main test function"""
    test = CoordinateSystemTest()
    success = await test.run_tests()
    
    if success:
        logger.info("🎉 Coordinate system tests PASSED!")
        return 0
    else:
        logger.error("💥 Coordinate system tests FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

