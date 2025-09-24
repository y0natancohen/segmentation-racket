#!/usr/bin/env python3
"""
Simple test to check segmentation model loading.
"""

import asyncio
import logging
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_send_recv', 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))

from video_segmentation_integration import VideoSegmentationProcessor, create_segmentation_args

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_segmentation():
    """Test segmentation model loading."""
    
    try:
        # Create segmentation args
        args = create_segmentation_args()
        logger.info(f"Created segmentation args: {args}")
        
        # Create processor
        processor = VideoSegmentationProcessor(args)
        logger.info("Created VideoSegmentationProcessor")
        
        # Initialize
        await processor.initialize()
        logger.info("✅ Segmentation processor initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_segmentation())
