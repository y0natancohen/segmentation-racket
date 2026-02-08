#!/usr/bin/env python3
"""
Simple camera test - captures and displays video from webcam.

Usage:
    python3 test_camera.py              # Use default camera (0)
    python3 test_camera.py --cam 1     # Use camera 1
    python3 test_camera.py --width 1280 --height 720  # Set resolution
"""

import argparse
import io
import os
import subprocess
import sys
import threading
import time

import cv2
from flask import Flask, Response, render_template_string

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
try:
    cv2.setLogLevel(0)  # 0 = SILENT
except AttributeError:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Simple camera capture and display test")
    p.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    p.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    p.add_argument("--force", action="store_true", help="Try to open camera even if it appears to be in use")
    p.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    return p.parse_args()


def check_camera_in_use(device_path):
    """Check if camera device is being used by another process."""
    try:
        result = subprocess.run(
            ['lsof', device_path],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse lsof output to find processes
            lines = result.stdout.strip().split('\n')
            processes = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    cmd = parts[0]
                    pid = parts[1]
                    processes.append((cmd, pid))
            return processes
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


# Global variables for camera and frame
camera = None
latest_frame = None
frame_lock = threading.Lock()
frame_count = 0
fps = 0.0

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Camera Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            background: #000;
            padding: 10px;
            border-radius: 8px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 4px;
        }
        .stats {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
        }
        .stat-item {
            display: inline-block;
            margin: 0 20px;
            font-size: 18px;
        }
        .stat-label {
            color: #888;
            font-size: 14px;
        }
        .stat-value {
            color: #4CAF50;
            font-weight: bold;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #45a049;
        }
        .info {
            text-align: center;
            color: #888;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📹 Camera Test</h1>
        
        <div class="video-container">
            <img id="video" src="/video_feed" alt="Camera Feed">
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0.0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Frames</div>
                <div class="stat-value" id="frames">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Resolution</div>
                <div class="stat-value" id="resolution">-</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="saveFrame()">💾 Save Frame</button>
            <button onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="info">
            Camera streaming at {{ host }}:{{ port }}
        </div>
    </div>
    
    <script>
        // Update stats periodically
        setInterval(function() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('frames').textContent = data.frames;
                    document.getElementById('resolution').textContent = data.resolution;
                })
                .catch(err => console.error('Error fetching stats:', err));
        }, 1000);
        
        function saveFrame() {
            fetch('/save_frame', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert('Frame saved: ' + data.filename);
                })
                .catch(err => {
                    alert('Error saving frame: ' + err);
                });
        }
    </script>
</body>
</html>
"""


def generate_frames():
    """Generator function for video streaming."""
    global latest_frame, frame_lock
    while True:
        with frame_lock:
            if latest_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


def camera_capture_loop(cap, args):
    """Background thread to capture frames from camera."""
    global latest_frame, frame_count, fps, frame_lock
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[test_camera] Failed to read frame, exiting capture loop")
            break
        
        frame_count += 1
        
        # Calculate FPS
        now = time.time()
        if now - last_time > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / (now - last_time))
        last_time = now
        
        # Add FPS text to frame
        fps_text = f"FPS: {fps:.1f} | Frame: {frame_count}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add resolution text
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res_text = f"Resolution: {actual_w}x{actual_h}"
        cv2.putText(frame, res_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update latest frame
        with frame_lock:
            latest_frame = frame.copy()
        
        # Print FPS every second
        if frame_count % 30 == 0:
            print(f"[test_camera] FPS: {fps:.1f}, Frames: {frame_count}")


def main():
    args = parse_args()
    
    # Check device file exists and permissions
    device_path = f"/dev/video{args.cam}"
    if os.path.exists(device_path):
        is_readable = os.access(device_path, os.R_OK)
        if not is_readable:
            print(f"[WARN] Device {device_path} exists but is not readable")
            print(f"[HINT] Try: sudo chmod 666 {device_path}")
            print(f"[HINT] Or add user to video group: sudo usermod -a -G video $USER")
    else:
        print(f"[WARN] Device {device_path} does not exist")
    
    # Check if camera is in use
    processes = check_camera_in_use(device_path)
    if processes and not args.force:
        print(f"\n[WARNING] Camera {device_path} is currently in use by:")
        for cmd, pid in processes:
            print(f"  - {cmd} (PID {pid})")
        print(f"\n[SOLUTIONS]")
        print(f"  1. Close the application using the camera (e.g., Chrome, Zoom, etc.)")
        print(f"  2. Kill the process: kill {processes[0][1]}")
        print(f"  3. Try a different camera: python3 test_camera.py --cam 1")
        print(f"  4. Force attempt anyway: python3 test_camera.py --force")
        print(f"\n[INFO] Checking for other available cameras...")
        
        # Check for other available cameras
        available_cams = []
        for i in range(5):
            if i == args.cam:
                continue
            test_path = f"/dev/video{i}"
            if os.path.exists(test_path):
                test_processes = check_camera_in_use(test_path)
                if not test_processes:
                    # Quick test if we can open it
                    test_cap = cv2.VideoCapture(i)
                    if test_cap.isOpened():
                        test_cap.release()
                        available_cams.append(i)
        
        if available_cams:
            print(f"  Found available cameras: {available_cams}")
            print(f"  Try: python3 test_camera.py --cam {available_cams[0]}")
        else:
            print(f"  No other cameras found or all are in use")
        
        sys.exit(1)
    
    if processes and args.force:
        print(f"[WARN] Camera is in use by {len(processes)} process(es), but --force specified, continuing...")
    
    print(f"[test_camera] Opening camera {args.cam}...")
    
    # Try to open camera with retries - try both index and device path
    cap = None
    max_retries = 3
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    # Try both camera index and device path
    camera_sources = [args.cam, device_path]
    
    for retry in range(max_retries):
        for camera_source in camera_sources:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_source, backend)
                    if cap.isOpened():
                        # Try a test read to verify camera works
                        test_ret, _ = cap.read()
                        if test_ret:
                            source_str = f"device {camera_source}" if isinstance(camera_source, str) else f"index {camera_source}"
                            print(f"[test_camera] Camera opened successfully ({source_str}, backend: {backend})")
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                            cap = None
                except Exception as e:
                    if cap:
                        cap.release()
                        cap = None
                    # Only print error if it's not the expected permission error
                    if "Permission denied" not in str(e):
                        print(f"[test_camera] Backend {backend} failed: {e}")
                
                if cap and cap.isOpened():
                    break
            
            if cap and cap.isOpened():
                break
        
        if cap and cap.isOpened():
            break
            
        if retry < max_retries - 1:
            time.sleep(0.5)
            print(f"[test_camera] Retrying ({retry + 1}/{max_retries})...")
    
    if not cap or not cap.isOpened():
        print(f"\n[ERROR] Failed to open camera {args.cam} after {max_retries} attempts")
        print(f"[DIAGNOSTICS]")
        print(f"  Device path: {device_path}")
        print(f"  Device exists: {os.path.exists(device_path)}")
        if os.path.exists(device_path):
            import stat
            st = os.stat(device_path)
            print(f"  Device permissions: {oct(st.st_mode)}")
            print(f"  Readable: {os.access(device_path, os.R_OK)}")
        print(f"\n[SOLUTIONS]")
        print(f"  1. Check if camera is in use: lsof {device_path}")
        print(f"  2. Fix permissions temporarily: sudo chmod 666 {device_path}")
        print(f"  3. Fix permissions permanently: sudo usermod -a -G video $USER")
        print(f"     (then log out and back in)")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[test_camera] Camera properties:")
    print(f"  Resolution: {actual_w}x{actual_h} (requested: {args.width}x{args.height})")
    print(f"  FPS: {actual_fps:.1f}")
    
    # Store camera globally
    global camera
    camera = cap
    
    # Create Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, host=args.host, port=args.port)
    
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/stats')
    def stats():
        global frame_count, fps, camera
        actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) if camera else 0
        actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) if camera else 0
        return {
            'fps': fps,
            'frames': frame_count,
            'resolution': f"{actual_w}x{actual_h}"
        }
    
    @app.route('/save_frame', methods=['POST'])
    def save_frame():
        global latest_frame, frame_lock
        with frame_lock:
            if latest_frame is not None:
                filename = f"camera_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, latest_frame)
                return {'success': True, 'filename': filename}
        return {'success': False, 'error': 'No frame available'}, 400
    
    # Start camera capture in background thread
    capture_thread = threading.Thread(target=camera_capture_loop, args=(cap, args), daemon=True)
    capture_thread.start()
    
    print(f"\n[test_camera] Web server starting...")
    print(f"[test_camera] Open your browser and go to: http://{args.host}:{args.port}")
    print(f"[test_camera] Press Ctrl+C to stop\n")
    
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[test_camera] Interrupted by user")
    finally:
        cap.release()
        print(f"[test_camera] Done - processed {frame_count} frames")


if __name__ == "__main__":
    main()
