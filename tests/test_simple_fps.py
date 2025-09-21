#!/usr/bin/env python3
"""
Simple FPS Test - Direct Python Generator Test
Tests the Python rectangle generator directly without the full pipeline.
"""

import asyncio
import time
import websockets
import json
import sys
import os

# Add the segmentation directory to the path so we can import polygon_generator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'segmentation'))

from polygon_generator import PolygonGenerator


async def test_direct_fps(duration_seconds: int = 5):
    """Test FPS directly with the Python generator."""
    print("🧪 Starting Direct FPS Test")
    print("=" * 40)
    
    # Create generator
    generator = PolygonGenerator(fps=60)
    print(f"✅ Created generator with FPS: {generator.fps}")
    
    # Start server
    print("🚀 Starting WebSocket server...")
    server_task = asyncio.create_task(generator.start_server())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    try:
        # Connect to WebSocket
        print("📡 Connecting to WebSocket...")
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected successfully")
            
            # Test FPS
            print(f"📊 Testing FPS for {duration_seconds} seconds...")
            start_time = time.time()
            message_count = 0
            last_timestamp = 0
            frame_times = []
            
            while time.time() - start_time < duration_seconds:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(message)
                    
                    # Calculate frame time
                    current_timestamp = data['timestamp']
                    if last_timestamp > 0:
                        frame_time = current_timestamp - last_timestamp
                        frame_times.append(frame_time)
                    
                    last_timestamp = current_timestamp
                    message_count += 1
                    
                except asyncio.TimeoutError:
                    continue
            
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_fps = message_count / actual_duration
            
            # Calculate statistics
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                min_frame_time = min(frame_times)
                max_frame_time = max(frame_times)
            else:
                avg_frame_time = min_frame_time = max_frame_time = 0
            
            # Print results
            print("\n" + "=" * 40)
            print("📊 FPS TEST RESULTS")
            print("=" * 40)
            print(f"Messages received: {message_count}")
            print(f"Test duration: {actual_duration:.2f}s")
            print(f"Actual FPS: {actual_fps:.2f}")
            print(f"Target FPS: 60.0")
            print(f"Performance: {(actual_fps/60.0)*100:.1f}% of target")
            print(f"Average frame time: {avg_frame_time*1000:.2f}ms")
            print(f"Min frame time: {min_frame_time*1000:.2f}ms")
            print(f"Max frame time: {max_frame_time*1000:.2f}ms")
            print(f"Target frame time: 16.67ms")
            
            # Check if we meet the requirement
            if actual_fps >= 59.0:
                print(f"\n✅ PASS: FPS >= 59 (achieved {actual_fps:.2f})")
                success = True
            else:
                print(f"\n❌ FAIL: FPS < 59 (achieved {actual_fps:.2f})")
                success = False
            
            print("=" * 40)
            return success
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        generator.running = False
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def main():
    """Main test function."""
    success = await test_direct_fps(duration_seconds=5)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
