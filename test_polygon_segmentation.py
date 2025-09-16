#!/usr/bin/env python3
"""
Test script for polygon segmentation functionality.
This script demonstrates how to use the new polygon features in segmentation.py
"""

import os
import sys
import subprocess
import time

def test_polygon_segmentation():
    """Test the polygon segmentation functionality"""
    print("🧪 Testing Polygon Segmentation Features")
    print("=" * 50)
    
    # Test 1: Check if segmentation script has polygon options
    print("\n1. Checking polygon options in segmentation script...")
    try:
        # Just check if the script can be imported and has the functions
        sys.path.append('.')
        import segmentation
        if hasattr(segmentation, 'matte_to_polygon') and hasattr(segmentation, 'draw_polygon_on_image'):
            print("✅ Polygon functions found in segmentation script")
        else:
            print("❌ Polygon functions not found")
            return False
    except Exception as e:
        print(f"❌ Error importing segmentation script: {e}")
        return False
    
    # Test 2: Test polygon function directly
    print("\n2. Testing matte_to_polygon function...")
    try:
        import numpy as np
        import cv2
        sys.path.append('.')
        from segmentation import matte_to_polygon, draw_polygon_on_image
        
        # Create a test segmentation map (circle)
        test_matte = np.zeros((200, 200), dtype=np.float32)
        cv2.circle(test_matte, (100, 100), 50, 1.0, -1)
        
        # Test polygon extraction
        polygon = matte_to_polygon(test_matte, threshold=0.5, min_area=1000)
        
        if polygon is not None and len(polygon) > 3:
            print(f"✅ Polygon extraction successful: {len(polygon)} vertices")
        else:
            print("❌ Polygon extraction failed")
            return False
            
        # Test polygon drawing
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image = draw_polygon_on_image(test_image, polygon, color=(0, 255, 0), thickness=2)
        
        if test_image is not None:
            print("✅ Polygon drawing successful")
        else:
            print("❌ Polygon drawing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing polygon functions: {e}")
        return False
    
    # Test 2.5: Test thresholded mask return
    print("\n2.5. Testing thresholded mask return...")
    try:
        # Test polygon extraction with mask return
        polygon, thresholded_mask = matte_to_polygon(test_matte, threshold=0.5, min_area=1000, return_mask=True)
        
        if polygon is not None and thresholded_mask is not None:
            print(f"✅ Thresholded mask return successful: {thresholded_mask.shape}")
        else:
            print("❌ Thresholded mask return failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing thresholded mask: {e}")
        return False
    
    # Test 3: Test with different polygon parameters
    print("\n3. Testing different polygon parameters...")
    try:
        # Test with different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            polygon = matte_to_polygon(test_matte, threshold=threshold, min_area=1000)
            if polygon is not None:
                print(f"✅ Threshold {threshold}: {len(polygon)} vertices")
            else:
                print(f"⚠️  Threshold {threshold}: No polygon found")
                
    except Exception as e:
        print(f"❌ Error testing parameters: {e}")
        return False
    
    print("\n🎉 All polygon segmentation tests passed!")
    print("\nUsage examples:")
    print("  # Save polygon images with live display")
    print("  python segmentation.py --save_polygon --show_polygon --live_display")
    print("  # Show thresholded segmentation map")
    print("  python segmentation.py --show_threshold --show_alpha --live_display")
    print("  # Show both polygon and threshold")
    print("  python segmentation.py --show_polygon --show_threshold --live_display")
    print("  # Save polygon images only (headless)")
    print("  python segmentation.py --save_polygon --headless --output_dir polygon_output")
    print("  # Adjust polygon extraction parameters")
    print("  python segmentation.py --save_polygon --polygon_threshold 0.3 --polygon_min_area 1000")
    
    return True

if __name__ == "__main__":
    success = test_polygon_segmentation()
    sys.exit(0 if success else 1)
