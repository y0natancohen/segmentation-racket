#!/usr/bin/env python3
"""
Full Pipeline FPS Test - Improved Version
Tests the complete dual-process architecture with better subprocess handling.
"""

import asyncio
import time
import websockets
import json
import subprocess
import signal
import os
import sys
import threading
from typing import List, Dict, Any


class FullPipelineFPSTest:
    """Test the full pipeline FPS performance with improved subprocess handling."""
    
    def __init__(self):
        self.python_process = None
        self.node_process = None
        self.received_messages: List[Dict[str, Any]] = []
        self.start_time = 0
        self.end_time = 0
        
    def start_python_generator_background(self):
        """Start Python generator in background thread."""
        def run_generator():
            try:
                # Import and run the generator directly
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from polygon_generator import main
                asyncio.run(main())
            except Exception as e:
                print(f"Python generator error: {e}")
        
        thread = threading.Thread(target=run_generator, daemon=True)
        thread.start()
        return thread
    
    async def test_websocket_fps(self, duration_seconds: int = 5) -> Dict[str, float]:
        """Test WebSocket FPS performance."""
        print(f"📡 Testing WebSocket FPS for {duration_seconds} seconds...")
        
        # Try to connect with retries
        websocket = None
        for attempt in range(10):  # More attempts
            try:
                print(f"   Attempt {attempt + 1}/10: Connecting to WebSocket...")
                websocket = await websockets.connect("ws://localhost:8765", timeout=3)
                print("   ✅ WebSocket connected successfully")
                break
            except Exception as e:
                print(f"   ❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < 9:
                    await asyncio.sleep(2)  # Longer wait between attempts
                else:
                    print("   ❌ All connection attempts failed")
                    return {}
        
        if not websocket:
            return {}
        
        try:
            self.start_time = time.time()
            message_count = 0
            last_timestamp = 0
            frame_times = []
            
            # Collect messages for specified duration
            while time.time() - self.start_time < duration_seconds:
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
                    self.received_messages.append(data)
                    
                except asyncio.TimeoutError:
                    continue
            
            self.end_time = time.time()
            actual_duration = self.end_time - self.start_time
            actual_fps = message_count / actual_duration
            
            # Calculate statistics
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                min_frame_time = min(frame_times)
                max_frame_time = max(frame_times)
                target_frame_time = 1.0 / 60.0  # 16.67ms for 60 FPS
            else:
                avg_frame_time = min_frame_time = max_frame_time = 0
                target_frame_time = 1.0 / 60.0
            
            return {
                'message_count': message_count,
                'duration': actual_duration,
                'actual_fps': actual_fps,
                'target_fps': 60.0,
                'fps_percentage': (actual_fps / 60.0) * 100,
                'avg_frame_time_ms': avg_frame_time * 1000,
                'min_frame_time_ms': min_frame_time * 1000,
                'max_frame_time_ms': max_frame_time * 1000,
                'target_frame_time_ms': target_frame_time * 1000,
                'frame_times': frame_times
            }
            
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            return {}
        finally:
            if websocket:
                await websocket.close()
    
    async def test_message_consistency(self) -> Dict[str, Any]:
        """Test message format and consistency."""
        print("🔍 Testing message consistency...")
        
        if not self.received_messages:
            return {'error': 'No messages received'}
        
        # Check message format
        required_fields = ['timestamp', 'position', 'velocity', 'phase', 'elapsed_time']
        format_errors = []
        
        for i, message in enumerate(self.received_messages[:10]):  # Check first 10 messages
            for field in required_fields:
                if field not in message:
                    format_errors.append(f"Message {i}: Missing field '{field}'")
        
        # Check timestamp consistency
        timestamps = [msg['timestamp'] for msg in self.received_messages]
        timestamp_errors = []
        
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                timestamp_errors.append(f"Non-increasing timestamp at message {i}")
        
        # Check position range
        positions = [msg['position']['y'] for msg in self.received_messages]
        min_y = min(positions)
        max_y = max(positions)
        expected_min = 240  # 330 - 90 (center_y - amplitude)
        expected_max = 420  # 330 + 90 (center_y + amplitude)
        
        range_errors = []
        if min_y < expected_min - 5 or min_y > expected_min + 5:
            range_errors.append(f"Min Y position {min_y} outside expected range {expected_min}±5")
        if max_y < expected_max - 5 or max_y > expected_max + 5:
            range_errors.append(f"Max Y position {max_y} outside expected range {expected_max}±5")
        
        return {
            'total_messages': len(self.received_messages),
            'format_errors': format_errors,
            'timestamp_errors': timestamp_errors,
            'range_errors': range_errors,
            'position_range': {'min': min_y, 'max': max_y, 'expected_min': expected_min, 'expected_max': expected_max}
        }
    
    async def run_full_test(self, duration_seconds: int = 5) -> Dict[str, Any]:
        """Run the complete FPS test pipeline."""
        print("🧪 Starting Full Pipeline FPS Test")
        print("=" * 50)
        
        try:
            # Start Python generator in background
            print("🚀 Starting Python rectangle generator...")
            generator_thread = self.start_python_generator_background()
            
            # Wait for server to start
            await asyncio.sleep(3)
            print("✅ Python generator started")
            
            # Test WebSocket FPS
            fps_results = await self.test_websocket_fps(duration_seconds)
            if not fps_results:
                return {'error': 'Failed to test WebSocket FPS'}
            
            # Test message consistency
            consistency_results = await self.test_message_consistency()
            
            # Combine results
            results = {
                'fps_test': fps_results,
                'consistency_test': consistency_results,
                'success': fps_results.get('actual_fps', 0) >= 59.0
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return {'error': str(e)}


def print_results(results: Dict[str, Any]):
    """Print test results in a formatted way."""
    print("\n" + "=" * 50)
    print("📊 FULL PIPELINE FPS TEST RESULTS")
    print("=" * 50)
    
    if 'error' in results:
        print(f"❌ Test failed: {results['error']}")
        return
    
    fps_test = results.get('fps_test', {})
    consistency_test = results.get('consistency_test', {})
    
    # FPS Results
    print(f"📈 FPS Performance:")
    print(f"   Messages received: {fps_test.get('message_count', 0)}")
    print(f"   Test duration: {fps_test.get('duration', 0):.2f}s")
    print(f"   Actual FPS: {fps_test.get('actual_fps', 0):.2f}")
    print(f"   Target FPS: {fps_test.get('target_fps', 60)}")
    print(f"   Performance: {fps_test.get('fps_percentage', 0):.1f}% of target")
    
    # Frame timing
    print(f"\n⏱️  Frame Timing:")
    print(f"   Average frame time: {fps_test.get('avg_frame_time_ms', 0):.2f}ms")
    print(f"   Min frame time: {fps_test.get('min_frame_time_ms', 0):.2f}ms")
    print(f"   Max frame time: {fps_test.get('max_frame_time_ms', 0):.2f}ms")
    print(f"   Target frame time: {fps_test.get('target_frame_time_ms', 16.67):.2f}ms")
    
    # Consistency Results
    print(f"\n🔍 Message Consistency:")
    print(f"   Total messages: {consistency_test.get('total_messages', 0)}")
    
    format_errors = consistency_test.get('format_errors', [])
    timestamp_errors = consistency_test.get('timestamp_errors', [])
    range_errors = consistency_test.get('range_errors', [])
    
    if format_errors:
        print(f"   ❌ Format errors: {len(format_errors)}")
        for error in format_errors[:3]:  # Show first 3 errors
            print(f"      - {error}")
    else:
        print(f"   ✅ Message format: OK")
    
    if timestamp_errors:
        print(f"   ❌ Timestamp errors: {len(timestamp_errors)}")
        for error in timestamp_errors[:3]:  # Show first 3 errors
            print(f"      - {error}")
    else:
        print(f"   ✅ Timestamp consistency: OK")
    
    if range_errors:
        print(f"   ❌ Range errors: {len(range_errors)}")
        for error in range_errors:
            print(f"      - {error}")
    else:
        print(f"   ✅ Position range: OK")
    
    # Overall result
    print(f"\n🎯 Overall Result:")
    if results.get('success', False):
        print(f"   ✅ PASS: FPS >= 59 (achieved {fps_test.get('actual_fps', 0):.2f})")
    else:
        print(f"   ❌ FAIL: FPS < 59 (achieved {fps_test.get('actual_fps', 0):.2f})")
    
    print("=" * 50)


async def main():
    """Main test function."""
    test = FullPipelineFPSTest()
    
    # Run test for 5 seconds
    results = await test.run_full_test(duration_seconds=5)
    
    # Print results
    print_results(results)
    
    # Exit with appropriate code
    if results.get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
