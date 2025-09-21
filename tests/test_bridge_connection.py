#!/usr/bin/env python3
"""
Simple test for the segmentation polygon bridge connection
Tests the WebSocket server without requiring camera or segmentation.
"""

import asyncio
import websockets
import json
import time
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation.segmentation_polygon_bridge import SegmentationPolygonBridge

async def test_bridge_connection():
    """Test the polygon bridge WebSocket connection"""
    print("🧪 Testing Segmentation Polygon Bridge Connection")
    print("=" * 50)
    
    # Create bridge
    bridge = SegmentationPolygonBridge(port=8765)
    
    # Start server in background
    server_task = asyncio.create_task(bridge.start_server())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    try:
        # Test WebSocket connection
        uri = "ws://localhost:8765"
        print(f"🔌 Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to bridge WebSocket")
            
            # Create test polygon data
            test_polygon = np.array([
                [100, 100],
                [200, 100], 
                [200, 200],
                [100, 200]
            ], dtype=np.float32)
            
            # Send test polygon data
            print("📤 Sending test polygon data...")
            await bridge.send_polygon_data(test_polygon, frame_size=(640, 480))
            
            # Wait for message
            print("📥 Waiting for message...")
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                print("✅ Received message:")
                print(f"   Position: {data['position']}")
                print(f"   Vertices: {len(data['vertices'])} points")
                print(f"   Rotation: {data['rotation']}")
                
                # Validate message format
                required_fields = ["position", "vertices", "rotation"]
                for field in required_fields:
                    if field not in data:
                        print(f"❌ Missing field: {field}")
                        return False
                
                if len(data["vertices"]) != 4:
                    print(f"❌ Expected 4 vertices, got {len(data['vertices'])}")
                    return False
                
                print("✅ Message format validation passed!")
                return True
                
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for message")
                return False
                
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False
    finally:
        # Cleanup
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        print("🧹 Cleanup completed")

async def main():
    """Main test function"""
    success = await test_bridge_connection()
    
    if success:
        print("🎉 Bridge connection test PASSED!")
        return 0
    else:
        print("💥 Bridge connection test FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
