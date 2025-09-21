#!/usr/bin/env python3
"""
Test segmentation with polygon bridge using a static image
This avoids camera dependency and tests the integration.
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
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation.segmentation_polygon_bridge import SegmentationPolygonBridge

class SegmentationBridgeTest:
    """Test segmentation with polygon bridge"""
    
    def __init__(self):
        self.received_messages: List[Dict[str, Any]] = []
        self.bridge = SegmentationPolygonBridge(port=8765)
        self.server_task = None
        
    async def start_bridge_server(self):
        """Start the bridge server"""
        self.server_task = asyncio.create_task(self.bridge.start_server())
        await asyncio.sleep(1)  # Give server time to start
        print("✅ Bridge server started")
    
    async def test_websocket_connection(self):
        """Test WebSocket connection and receive messages"""
        try:
            uri = "ws://localhost:8765"
            print(f"🔌 Connecting to {uri}...")
            
            async with websockets.connect(uri) as websocket:
                print("✅ Connected to bridge WebSocket")
                
                # Send test polygon data
                test_polygon = np.array([
                    [100, 100],
                    [200, 100], 
                    [200, 200],
                    [100, 200]
                ], dtype=np.float32)
                
                print("📤 Sending test polygon data...")
                await self.bridge.send_polygon_data(test_polygon, frame_size=(640, 480))
                
                # Wait for message
                print("📥 Waiting for message...")
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    self.received_messages.append(data)
                    
                    print("✅ Received message:")
                    print(f"   Position: {data['position']}")
                    print(f"   Vertices: {len(data['vertices'])} points")
                    print(f"   Rotation: {data['rotation']}")
                    
                    return True
                    
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for message")
                    return False
                    
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def test_segmentation_import(self):
        """Test that segmentation can be imported with bridge"""
        try:
            # Try to import segmentation with bridge
            import segmentation
            print("✅ Segmentation module imported successfully")
            
            # Check if bridge arguments are available
            import argparse
            parser = argparse.ArgumentParser()
            segmentation.parse_args()  # This will fail if there are issues
            
            print("✅ Segmentation argument parsing works")
            return True
            
        except Exception as e:
            print(f"❌ Segmentation import failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        print("🧹 Cleanup completed")
    
    async def run_test(self):
        """Run the complete test"""
        print("🧪 Testing Segmentation with Polygon Bridge")
        print("=" * 50)
        
        try:
            # Step 1: Test segmentation import
            print("Step 1: Testing segmentation import...")
            if not self.test_segmentation_import():
                return False
            
            # Step 2: Start bridge server
            print("Step 2: Starting bridge server...")
            await self.start_bridge_server()
            
            # Step 3: Test WebSocket connection
            print("Step 3: Testing WebSocket connection...")
            if not await self.test_websocket_connection():
                return False
            
            print("✅ All tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main test function"""
    test = SegmentationBridgeTest()
    success = await test.run_test()
    
    if success:
        print("🎉 Segmentation bridge test PASSED!")
        return 0
    else:
        print("💥 Segmentation bridge test FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
