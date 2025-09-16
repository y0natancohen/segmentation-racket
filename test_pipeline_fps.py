#!/usr/bin/env python3
"""
Pipeline FPS Test - Using Existing Infrastructure
Tests FPS using the existing startup script infrastructure.
"""

import asyncio
import time
import websockets
import json
import subprocess
import sys
import os


async def test_pipeline_fps(duration_seconds: int = 5):
    """Test FPS using the existing startup script."""
    print("🧪 Starting Pipeline FPS Test")
    print("=" * 40)
    
    # Start the game using the existing script
    print("🚀 Starting dual-process architecture...")
    startup_script = "./start_game.sh"
    
    if not os.path.exists(startup_script):
        print(f"❌ Startup script {startup_script} not found")
        return False
    
    # Start the processes
    process = subprocess.Popen(
        [startup_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for processes to start
    print("⏳ Waiting for processes to start...")
    await asyncio.sleep(5)
    
    try:
        # Test WebSocket connection
        print("📡 Testing WebSocket FPS...")
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected to WebSocket")
            
            # Test FPS
            start_time = time.time()
            message_count = 0
            last_rotation = 0
            frame_times = []
            
            while time.time() - start_time < duration_seconds:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(message)
                    
                    # Verify message format
                    if 'position' not in data or 'vertices' not in data or 'rotation' not in data:
                        print("Missing required fields in message")
                        return False
                    
                    # Calculate frame time using rotation (0.5 rad/s = 0.5 rad per second)
                    current_rotation = data['rotation']
                    if last_rotation > 0:
                        rotation_diff = current_rotation - last_rotation
                        # Convert rotation difference to time (0.5 rad/s)
                        frame_time = rotation_diff / 0.5
                        frame_times.append(frame_time)
                    
                    last_rotation = current_rotation
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
            print("📊 PIPELINE FPS TEST RESULTS")
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
        print("🧹 Cleaning up processes...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception as e:
            print(f"Warning: Error stopping processes: {e}")
        
        # Additional cleanup
        try:
            subprocess.run(["pkill", "-f", "rectangle_generator.py"], check=False)
            subprocess.run(["pkill", "-f", "npm run dev"], check=False)
            subprocess.run(["pkill", "-f", "vite"], check=False)
        except Exception as e:
            print(f"Warning: Error in additional cleanup: {e}")


async def main():
    """Main test function."""
    success = await test_pipeline_fps(duration_seconds=5)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
