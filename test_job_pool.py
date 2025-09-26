#!/usr/bin/env python3
"""
Test script to demonstrate the job pool functionality.
"""

import asyncio
import time
import logging
from segmentation.video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_job_pool():
    """Test the job pool functionality."""
    print("=== Testing Job Pool Functionality ===")
    
    # Create processor
    args = create_segmentation_args()
    processor = VideoSegmentationProcessor(args)
    
    # Set max concurrent jobs to 2
    processor.set_max_concurrent_jobs(2)
    
    # Simulate rapid frame arrivals
    print("Simulating rapid frame arrivals...")
    
    for i in range(10):
        frame_data = {
            'frame': None,  # We're not actually processing, just testing the job pool
            'timestamp': time.time(),
            'connection_id': 'test_conn'
        }
        
        # This will trigger the job pool logic
        processor._process_frame(frame_data)
        
        # Check status
        status = processor.get_job_pool_status()
        print(f"Frame {i}: Active jobs: {status['active_jobs']}/{status['max_concurrent_jobs']}, "
              f"Dropped: {status['dropped_frames']}")
        
        # Small delay to simulate frame arrival
        await asyncio.sleep(0.1)
    
    print(f"\nFinal status: {processor.get_job_pool_status()}")
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_job_pool())
