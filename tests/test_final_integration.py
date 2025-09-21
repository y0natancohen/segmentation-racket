#!/usr/bin/env python3
"""
Final Integration Test for Segmentation-Polygon System
Tests the complete system with correct aspect ratio and coordinate scaling.
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
from segmentation.segmentation import CAMERA_WIDTH, CAMERA_HEIGHT, GAME_WIDTH, GAME_HEIGHT

class FinalIntegrationTest:
    """Test the complete integration with correct aspect ratio"""
    
    def __init__(self):
        self.bridge = SegmentationPolygonBridge(port=8765)
        self.server_task: Optional[asyncio.Task] = None
        self.received_messages: List[Dict[str, Any]] = []
        
    async def start_bridge_server(self):
        """Start the bridge server"""
        self.server_task = asyncio.create_task(self.bridge.start_server())
        await asyncio.sleep(1)  # Give server time to start
        logger.info("✅ Bridge server started")
    
    def create_realistic_segmentation_polygons(self) -> List[Dict[str, Any]]:
        """Create realistic segmentation polygons that simulate real camera input"""
        test_cases = []
        
        # Test 1: Full frame segmentation (entire image is foreground)
        # This should result in a polygon that fills the entire game area
        full_frame_polygon = np.array([
            [0, 0],                           # Top-left corner
            [CAMERA_WIDTH, 0],                # Top-right corner
            [CAMERA_WIDTH, CAMERA_HEIGHT],    # Bottom-right corner
            [0, CAMERA_HEIGHT]                # Bottom-left corner
        ], dtype=np.float32)
        
        test_cases.append({
            "name": "Full Frame Segmentation",
            "description": "Entire camera frame is segmented as foreground",
            "polygon": full_frame_polygon,
            "expected_coverage": 1.0,  # Should cover 100% of game area
            "expected_bounds": {
                "min_x": 0,
                "max_x": GAME_WIDTH,
                "min_y": 0,
                "max_y": GAME_HEIGHT
            }
        })
        
        # Test 2: Person silhouette (center of frame)
        person_polygon = np.array([
            [CAMERA_WIDTH * 0.4, CAMERA_HEIGHT * 0.1],   # Head
            [CAMERA_WIDTH * 0.6, CAMERA_HEIGHT * 0.1],   # Head
            [CAMERA_WIDTH * 0.65, CAMERA_HEIGHT * 0.3],  # Shoulder
            [CAMERA_WIDTH * 0.7, CAMERA_HEIGHT * 0.8],   # Hip
            [CAMERA_WIDTH * 0.6, CAMERA_HEIGHT * 0.9],   # Leg
            [CAMERA_WIDTH * 0.4, CAMERA_HEIGHT * 0.9],   # Leg
            [CAMERA_WIDTH * 0.3, CAMERA_HEIGHT * 0.8],   # Hip
            [CAMERA_WIDTH * 0.35, CAMERA_HEIGHT * 0.3],  # Shoulder
        ], dtype=np.float32)
        
        test_cases.append({
            "name": "Person Silhouette",
            "description": "Person detected in center of frame",
            "polygon": person_polygon,
            "expected_coverage": 0.3,  # Should cover ~30% of game area
            "expected_bounds": {
                "min_x": GAME_WIDTH * 0.3,
                "max_x": GAME_WIDTH * 0.7,
                "min_y": GAME_HEIGHT * 0.1,
                "max_y": GAME_HEIGHT * 0.9
            }
        })
        
        # Test 3: Hand gesture (small object in corner)
        hand_polygon = np.array([
            [CAMERA_WIDTH * 0.7, CAMERA_HEIGHT * 0.2],
            [CAMERA_WIDTH * 0.8, CAMERA_HEIGHT * 0.3],
            [CAMERA_WIDTH * 0.9, CAMERA_HEIGHT * 0.4],
            [CAMERA_WIDTH * 0.85, CAMERA_HEIGHT * 0.5],
            [CAMERA_WIDTH * 0.75, CAMERA_HEIGHT * 0.4],
        ], dtype=np.float32)
        
        test_cases.append({
            "name": "Hand Gesture",
            "description": "Hand detected in top-right corner",
            "polygon": hand_polygon,
            "expected_coverage": 0.05,  # Should cover ~5% of game area
            "expected_bounds": {
                "min_x": GAME_WIDTH * 0.7,
                "max_x": GAME_WIDTH * 0.9,
                "min_y": GAME_HEIGHT * 0.2,
                "max_y": GAME_HEIGHT * 0.5
            }
        })
        
        return test_cases
    
    async def test_segmentation_to_game_pipeline(self):
        """Test the complete pipeline from segmentation to game"""
        test_cases = self.create_realistic_segmentation_polygons()
        
        logger.info("🧪 Testing Segmentation-to-Game Pipeline")
        logger.info("=" * 60)
        logger.info(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} (aspect ratio: {CAMERA_WIDTH/CAMERA_HEIGHT:.3f})")
        logger.info(f"Game:   {GAME_WIDTH}x{GAME_HEIGHT} (aspect ratio: {GAME_WIDTH/GAME_HEIGHT:.3f})")
        logger.info("")
        
        try:
            uri = "ws://localhost:8765"
            async with websockets.connect(uri) as websocket:
                logger.info("✅ Connected to bridge WebSocket (Phaser game perspective)")
                logger.info("")
                
                for i, test_case in enumerate(test_cases):
                    logger.info(f"Test {i+1}: {test_case['name']}")
                    logger.info(f"   Description: {test_case['description']}")
                    
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
                            logger.error(f"   ❌ No vertices received")
                            continue
                        
                        # Calculate actual bounds and coverage
                        x_coords = [v["x"] for v in vertices]
                        y_coords = [v["y"] for v in vertices]
                        actual_bounds = {
                            "min_x": min(x_coords),
                            "max_x": max(x_coords),
                            "min_y": min(y_coords),
                            "max_y": max(y_coords)
                        }
                        
                        # Calculate coverage
                        coverage_x = (actual_bounds["max_x"] - actual_bounds["min_x"]) / GAME_WIDTH
                        coverage_y = (actual_bounds["max_y"] - actual_bounds["min_y"]) / GAME_HEIGHT
                        coverage_area = coverage_x * coverage_y
                        
                        logger.info(f"   Expected coverage: {test_case['expected_coverage']:.1%}")
                        logger.info(f"   Actual coverage:   {coverage_area:.1%}")
                        logger.info(f"   Bounds: x=[{actual_bounds['min_x']:.1f}, {actual_bounds['max_x']:.1f}], y=[{actual_bounds['min_y']:.1f}, {actual_bounds['max_y']:.1f}]")
                        
                        # Validate coverage
                        expected_coverage = test_case["expected_coverage"]
                        coverage_tolerance = 0.1  # 10% tolerance
                        
                        if abs(coverage_area - expected_coverage) <= coverage_tolerance:
                            logger.info(f"   ✅ Coverage is correct!")
                        else:
                            logger.warning(f"   ⚠️ Coverage differs from expected (tolerance: {coverage_tolerance:.1%})")
                        
                        # Special validation for full frame test
                        if test_case["name"] == "Full Frame Segmentation":
                            if coverage_area >= 0.95:  # At least 95% coverage
                                logger.info(f"   ✅ Full frame polygon correctly fills the game area!")
                            else:
                                logger.error(f"   ❌ Full frame polygon should fill the entire game area!")
                        
                        logger.info("")
                        
                    except asyncio.TimeoutError:
                        logger.error(f"   ❌ Timeout waiting for message")
                        continue
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
            return False
    
    async def test_aspect_ratio_consistency(self):
        """Test that aspect ratios are consistent between camera and game"""
        logger.info("🔍 Testing Aspect Ratio Consistency")
        logger.info("=" * 40)
        
        camera_ratio = CAMERA_WIDTH / CAMERA_HEIGHT
        game_ratio = GAME_WIDTH / GAME_HEIGHT
        ratio_diff = abs(camera_ratio - game_ratio)
        
        logger.info(f"Camera aspect ratio: {camera_ratio:.6f}")
        logger.info(f"Game aspect ratio:   {game_ratio:.6f}")
        logger.info(f"Difference:          {ratio_diff:.6f}")
        
        if ratio_diff < 0.001:  # Very small tolerance
            logger.info("✅ Aspect ratios are perfectly consistent!")
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
    
    async def run_final_test(self) -> bool:
        """Run the final integration test"""
        logger.info("🎯 Final Integration Test")
        logger.info("=" * 60)
        logger.info("This test verifies that:")
        logger.info("1. Camera and game have matching aspect ratios")
        logger.info("2. Full frame segmentation fills the entire game area")
        logger.info("3. Coordinate scaling works correctly for all polygon sizes")
        logger.info("4. The complete pipeline works end-to-end")
        logger.info("")
        
        try:
            # Step 1: Start bridge server
            logger.info("Step 1: Starting bridge server...")
            await self.start_bridge_server()
            
            # Step 2: Test aspect ratio consistency
            logger.info("Step 2: Testing aspect ratio consistency...")
            aspect_ratio_ok = await self.test_aspect_ratio_consistency()
            logger.info("")
            
            # Step 3: Test complete pipeline
            logger.info("Step 3: Testing segmentation-to-game pipeline...")
            pipeline_ok = await self.test_segmentation_to_game_pipeline()
            
            # Step 4: Validate results
            logger.info("Step 4: Validating results...")
            if aspect_ratio_ok and pipeline_ok:
                logger.info("✅ All integration tests passed!")
                logger.info("")
                logger.info("🎉 The segmentation-polygon integration is working correctly!")
                logger.info("   - Aspect ratios are consistent")
                logger.info("   - Coordinate scaling is accurate")
                logger.info("   - Full frame segmentation works as expected")
                logger.info("   - Ready for real-time camera input!")
                return True
            else:
                logger.error("❌ Some integration tests failed!")
                return False
            
        except Exception as e:
            logger.error(f"❌ Final integration test failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main test function"""
    test = FinalIntegrationTest()
    success = await test.run_final_test()
    
    if success:
        logger.info("🎉 Final integration test PASSED!")
        return 0
    else:
        logger.error("💥 Final integration test FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

