#!/usr/bin/env python3
"""
Debug Segmentation — visual debugger for the segmentation pipeline.

Reads from the local webcam, encodes each frame as JPEG, and feeds it
through the **exact same** SegmentationSession.process_frame() that the
WebSocket server uses.  Displays four panels in an OpenCV window:

  1. Original frame
  2. Alpha-matte heatmap (JET colourmap)
  3. Thresholded binary mask
  4. Original frame with polygon overlay

Also prints per-frame timing breakdown and polygon vertex count.

Usage:
    python debug_segmentation.py                       # defaults
    python debug_segmentation.py --cam 1 --dsr 0.25   # alternate camera
    python debug_segmentation.py --save_dir debug_out  # save frames to disk
"""

import argparse
import io
import json
import os
import sys
import threading
import time
import warnings

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string

from segmentation_server import (       # noqa: E402
    SegmentationSession,
    load_model,
    to_torch_image,
)
from segmentation.segmentation import (  # noqa: E402
    matte_to_polygon,
    draw_polygon_on_image,
    TimingStats,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Visual debugger for the segmentation pipeline",
    )
    p.add_argument("--cam", type=int, default=0, help="Camera index")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=360, help="Capture height")
    p.add_argument("--model_path", default="models/rvm_mobilenetv3.pth")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dsr", type=float, default=0.25, help="RVM downsample ratio")
    p.add_argument("--polygon_threshold", type=float, default=0.5)
    p.add_argument("--polygon_min_area", type=int, default=2000)
    p.add_argument("--polygon_epsilon", type=float, default=0.002)
    p.add_argument("--jpeg_quality", type=int, default=70,
                    help="JPEG encode quality 0-100 (matches browser quality)")
    p.add_argument("--save_dir", type=str, default=None,
                    help="If set, save debug panels to this directory")
    p.add_argument("--headless", action="store_true",
                    help="No GUI windows — print stats only (for SSH)")
    p.add_argument("--port", type=int, default=5001, help="Web server port (default: 5001)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    return p.parse_args()


# Global variables for sharing between threads
latest_combined_frame = None
latest_stats = {}
frame_lock = threading.Lock()
frame_idx_global = 0
frame_idx = 0
fps = 0.0
last_time = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Debug Segmentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            font-size: 14px;
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
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-label {
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #4CAF50;
            font-weight: bold;
            font-size: 18px;
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
        .panel-labels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 12px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Debug Segmentation</h1>
        <div class="subtitle">Four-panel visualization: Original | Alpha Matte | Mask | Polygon Overlay</div>
        
        <div class="video-container">
            <div class="panel-labels">
                <div>Top Left: Original Frame</div>
                <div>Top Right: Alpha Matte (JET colormap)</div>
                <div>Bottom Left: Thresholded Mask</div>
                <div>Bottom Right: Polygon Overlay</div>
            </div>
            <img id="video" src="/video_feed" alt="Debug Segmentation Feed">
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0.0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Frame</div>
                <div class="stat-value" id="frame">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Time</div>
                <div class="stat-value" id="total_ms">0ms</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Result</div>
                <div class="stat-value" id="result">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Vertices</div>
                <div class="stat-value" id="vertices">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Alpha Min</div>
                <div class="stat-value" id="alpha_min">0.000</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Alpha Max</div>
                <div class="stat-value" id="alpha_max">0.000</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Alpha Mean</div>
                <div class="stat-value" id="alpha_mean">0.000</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Decode</div>
                <div class="stat-value" id="decode_ms">0.0ms</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Inference</div>
                <div class="stat-value" id="infer_ms">0.0ms</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Polygon</div>
                <div class="stat-value" id="poly_ms">0.0ms</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Threshold</div>
                <div class="stat-value" id="threshold">0.5</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="saveFrame()">💾 Save Frame</button>
            <button onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="info">
            Debug segmentation streaming at {{ host }}:{{ port }}<br>
            Press Ctrl+C in terminal to stop
        </div>
    </div>
    
    <script>
        // Update stats periodically
        setInterval(function() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('frame').textContent = data.frame;
                    document.getElementById('total_ms').textContent = data.total_ms.toFixed(0) + 'ms';
                    document.getElementById('result').textContent = data.result;
                    document.getElementById('vertices').textContent = data.vertices;
                    document.getElementById('alpha_min').textContent = data.alpha_min.toFixed(3);
                    document.getElementById('alpha_max').textContent = data.alpha_max.toFixed(3);
                    document.getElementById('alpha_mean').textContent = data.alpha_mean.toFixed(3);
                    document.getElementById('decode_ms').textContent = data.decode_ms.toFixed(1) + 'ms';
                    document.getElementById('infer_ms').textContent = data.infer_ms.toFixed(1) + 'ms';
                    document.getElementById('poly_ms').textContent = data.poly_ms.toFixed(1) + 'ms';
                    document.getElementById('threshold').textContent = data.threshold;
                })
                .catch(err => console.error('Error fetching stats:', err));
        }, 500);
        
        function saveFrame() {
            fetch('/save_frame', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Frame saved: ' + data.filename);
                    } else {
                        alert('Error: ' + data.error);
                    }
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
    global latest_combined_frame, frame_lock
    while True:
        with frame_lock:
            if latest_combined_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', latest_combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


def main():
    args = parse_args()

    # ---- check camera availability -----------------------------------------
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"startup","hypothesisId":"B","location":"debug_segmentation.py:67","message":"Checking camera device permissions","data":{"requested_camera_index":args.cam},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    # Check if /dev/video* devices exist and are accessible
    import glob
    import stat
    import grp
    video_devices = glob.glob('/dev/video*')
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"startup","hypothesisId":"B","location":"debug_segmentation.py:72","message":"Video devices found","data":{"video_devices":video_devices},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    if video_devices:
        print(f"[debug_seg] Found video devices: {video_devices}")
        # Check if we can read the device
        target_device = f"/dev/video{args.cam}"
        if target_device in video_devices:
            try:
                st = os.stat(target_device)
                is_readable = os.access(target_device, os.R_OK)
                # #region agent log
                with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"runId":"startup","hypothesisId":"B","location":"debug_segmentation.py:82","message":"Device permissions check","data":{"device":target_device,"is_readable":is_readable,"mode":oct(st.st_mode)},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                if not is_readable:
                    print(f"[ERROR] Cannot read {target_device} - permission denied")
                    # Check if user is in video group
                    try:
                        video_gid = grp.getgrnam('video').gr_gid
                        user_gids = os.getgroups()
                        in_video_group = video_gid in user_gids
                    except (KeyError, AttributeError):
                        in_video_group = False
                    
                    if not in_video_group:
                        print(f"[HINT] User not in 'video' group. To fix:")
                        print(f"      sudo usermod -a -G video $USER")
                        print(f"      (then log out and back in)")
                    print(f"[HINT] Or temporarily: sudo chmod 666 {target_device}")
                    print(f"[WARN] Continuing anyway - camera might still work...")
            except Exception as e:
                # #region agent log
                with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"runId":"startup","hypothesisId":"B","location":"debug_segmentation.py:89","message":"Device stat failed","data":{"device":target_device,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                print(f"[WARN] Could not check permissions for {target_device}: {e}")
    else:
        print("[WARN] No /dev/video* devices found")
        # #region agent log
        with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"runId":"startup","hypothesisId":"A","location":"debug_segmentation.py:94","message":"No video devices found","data":{},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion

    # Try to detect available cameras via OpenCV (with timeout to avoid hanging)
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"startup","hypothesisId":"A","location":"debug_segmentation.py:98","message":"Checking camera availability via OpenCV","data":{"requested_camera_index":args.cam},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    available_cameras = []
    # Skip camera detection if we have permission issues - just try to open directly
    target_device = f"/dev/video{args.cam}"
    has_permission = os.path.exists(target_device) and os.access(target_device, os.R_OK)
    
    if has_permission:
        # Only check a few cameras quickly, with short timeouts
        for i in range(min(3, args.cam + 2)):  # Check up to requested cam + 2
            try:
                test_cap = cv2.VideoCapture(i)
                # Set a short timeout for the check
                test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if test_cap.isOpened():
                    # Try a quick read with timeout simulation
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)  # Small size for quick test
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                    available_cameras.append(i)
                test_cap.release()
                time.sleep(0.05)  # Short delay
            except Exception:
                pass
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"startup","hypothesisId":"A","location":"debug_segmentation.py:106","message":"Available cameras detected","data":{"available_cameras":available_cameras,"requested_index":args.cam},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    if available_cameras:
        print(f"[debug_seg] Available cameras: {available_cameras}")
        if args.cam not in available_cameras:
            print(f"[WARN] Camera index {args.cam} not in available list. Trying anyway...")
    else:
        if has_permission:
            print("[WARN] No cameras detected via OpenCV. Trying requested index anyway...")
        else:
            print("[WARN] Skipping camera detection due to permission issues. Will try to open camera directly...")
    
    # Small delay to ensure cameras are fully released
    if available_cameras:
        time.sleep(0.3)

    # ---- device selection --------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[debug_seg] Device: {device}")

    # ---- load model --------------------------------------------------------
    model = load_model(args.model_path, device, args.fp16)

    # ---- create session (the SAME class the server uses) -------------------
    session = SegmentationSession(
        model=model,
        device=device,
        fp16=args.fp16,
        dsr=args.dsr,
        polygon_threshold=args.polygon_threshold,
        polygon_min_area=args.polygon_min_area,
        polygon_epsilon=args.polygon_epsilon,
    )

    # ---- open camera -------------------------------------------------------
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"camera_init","hypothesisId":"A","location":"debug_segmentation.py:95","message":"Attempting to open camera","data":{"camera_index":args.cam,"requested_width":args.width,"requested_height":args.height},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    # Try opening camera with retries and different backends
    cap = None
    max_retries = 3
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]  # Try V4L2 first, then any available
    
    for retry in range(max_retries):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(args.cam, backend)
                # #region agent log
                is_opened = cap.isOpened()
                with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"runId":"camera_init","hypothesisId":"A","location":"debug_segmentation.py:98","message":"Camera open result","data":{"is_opened":is_opened,"camera_index":args.cam,"retry":retry,"backend":str(backend)},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                if is_opened:
                    # Try a test read to verify camera actually works
                    test_ret, _ = cap.read()
                    if test_ret:
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    if cap:
                        cap.release()
                        cap = None
            except Exception as e:
                print(f"[debug_seg] Backend {backend} failed: {e}")
                if cap:
                    cap.release()
                    cap = None
        
        if cap and cap.isOpened():
            break
            
        if retry < max_retries - 1:
            time.sleep(0.5)  # Wait before retry
            print(f"[debug_seg] Camera open failed, retrying ({retry + 1}/{max_retries})...")
    
    if not cap or not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {args.cam} after {max_retries} attempts")
        print(f"[HINT] Make sure the camera is not in use by another application")
        print(f"[HINT] Try: sudo chmod 666 /dev/video{args.cam} or add user to video group")
        # #region agent log
        with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"runId":"camera_init","hypothesisId":"A","location":"debug_segmentation.py:102","message":"Camera open FAILED","data":{"camera_index":args.cam},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        sys.exit(1)
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"camera_init","hypothesisId":"D","location":"debug_segmentation.py:105","message":"Setting camera properties","data":{"width":args.width,"height":args.height},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # #region agent log
    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"runId":"camera_init","hypothesisId":"C","location":"debug_segmentation.py:112","message":"Camera dimensions after set","data":{"actual_width":actual_w,"actual_height":actual_h,"requested_width":args.width,"requested_height":args.height},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    print(f"[debug_seg] Camera opened: {actual_w}x{actual_h}")
    if actual_w == 0 or actual_h == 0:
        print(f"[ERROR] Camera reports invalid dimensions {actual_w}x{actual_h}")
        # #region agent log
        with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"runId":"camera_init","hypothesisId":"C","location":"debug_segmentation.py:117","message":"Camera dimensions are 0x0 - attempting test read","data":{"actual_width":actual_w,"actual_height":actual_h},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        # Try a test read to see if camera actually works
        # #region agent log
        with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"runId":"camera_init","hypothesisId":"E","location":"debug_segmentation.py:120","message":"Attempting test read before dimensions check","data":{},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        test_ret, test_frame = cap.read()
        # #region agent log
        with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"runId":"camera_init","hypothesisId":"E","location":"debug_segmentation.py:123","message":"Test read result","data":{"read_success":test_ret,"frame_shape":list(test_frame.shape) if test_ret and test_frame is not None else None},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        if test_ret and test_frame is not None:
            actual_w, actual_h = test_frame.shape[1], test_frame.shape[0]
            # #region agent log
            with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"runId":"camera_init","hypothesisId":"E","location":"debug_segmentation.py:127","message":"Test read succeeded - using frame dimensions","data":{"actual_width":actual_w,"actual_height":actual_h},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            print(f"[debug_seg] Test read successful, actual dimensions: {actual_w}x{actual_h}")
        else:
            print(f"[ERROR] Test read also failed")
            # #region agent log
            with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"runId":"camera_init","hypothesisId":"E","location":"debug_segmentation.py:132","message":"Test read FAILED","data":{},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            cap.release()
            sys.exit(1)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- extra timing stats for the debug-only visualisation parts ---------
    viz_timing = TimingStats(max_samples=60)
    # Initialize global variables
    global frame_idx, fps, last_time
    frame_idx = 0
    fps = 0.0
    last_time = time.time()

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
        global latest_stats
        with frame_lock:
            stats_copy = latest_stats.copy()
        return stats_copy
    
    @app.route('/save_frame', methods=['POST'])
    def save_frame():
        global latest_combined_frame, frame_lock, frame_idx_global
        with frame_lock:
            if latest_combined_frame is not None:
                if args.save_dir:
                    filename = os.path.join(args.save_dir, f"debug_{frame_idx_global:06d}.jpg")
                else:
                    filename = f"debug_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, latest_combined_frame)
                return {'success': True, 'filename': filename}
        return {'success': False, 'error': 'No frame available'}, 400

    # Background thread for camera capture and processing
    def camera_processing_loop():
        global latest_combined_frame, latest_stats, frame_idx_global, frame_lock, frame_idx, fps, last_time
        
        try:
            while True:
                loop_start = time.time()

                # #region agent log
                with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"runId":"frame_loop","hypothesisId":"C","location":"debug_segmentation.py:145","message":"Attempting frame read","data":{"frame_idx":frame_idx},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                ret, frame = cap.read()
                # #region agent log
                with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"runId":"frame_loop","hypothesisId":"C","location":"debug_segmentation.py:148","message":"Frame read result","data":{"read_success":ret,"frame_shape":list(frame.shape) if ret and frame is not None else None},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                if not ret:
                    print("[debug_seg] Camera read failed, exiting")
                    # #region agent log
                    with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"runId":"frame_loop","hypothesisId":"C","location":"debug_segmentation.py:152","message":"Frame read FAILED - exiting loop","data":{"frame_idx":frame_idx},"timestamp":int(time.time()*1000)}) + '\n')
                    # #endregion
                    break

                # ---- JPEG encode (mimics what the browser does) ----------------
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                ok, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    print("[debug_seg] JPEG encode failed, skipping frame")
                    continue
                jpeg_bytes = jpeg_buf.tobytes()

                # ---- run the production process_frame --------------------------
                result = session.process_frame(jpeg_bytes)

                # ---- also run matte_to_polygon with return_mask for viz --------
                # We need the raw alpha matte, so replicate the decode+inference
                # from the session internals.  This is a second inference — we
                # accept the cost for debugging.  To avoid it, we expose the
                # alpha from the session directly:
                viz_start = time.time()

                # Re-decode the same JPEG for the visualization path
                buf_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame_for_viz = cv2.imdecode(buf_np, cv2.IMREAD_COLOR)

                # Run model again to get the alpha matte (for visualisation)
                src_t = to_torch_image(frame_for_viz, device, args.fp16)
                with torch.inference_mode():
                    _fgr, pha_t, *_ = model(src_t, *session.rec, args.dsr)
                pha_np = pha_t[0, 0].cpu().numpy()  # float32 [0,1]

                # Get polygon + thresholded mask
                poly_result = matte_to_polygon(
                    pha_np,
                    threshold=args.polygon_threshold,
                    min_area=args.polygon_min_area,
                    epsilon_ratio=args.polygon_epsilon,
                    return_mask=True,
                )
                if isinstance(poly_result, tuple):
                    polygon_viz, mask_viz = poly_result
                else:
                    polygon_viz = poly_result
                    mask_viz = None

                # ---- build the 4-panel display ---------------------------------
                h, w = frame.shape[:2]
                panel_h, panel_w = h, w

                # Panel 1: original frame
                p1 = frame.copy()
                cv2.putText(p1, "Original", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Panel 2: alpha matte heatmap
                pha_u8 = (pha_np * 255).astype(np.uint8)
                pha_resized = cv2.resize(pha_u8, (panel_w, panel_h))
                p2 = cv2.applyColorMap(pha_resized, cv2.COLORMAP_JET)
                stats_text = f"min={pha_np.min():.3f} max={pha_np.max():.3f} mean={pha_np.mean():.3f}"
                cv2.putText(p2, "Alpha Matte", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(p2, stats_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Panel 3: thresholded mask
                if mask_viz is not None:
                    mask_resized = cv2.resize(mask_viz, (panel_w, panel_h))
                    p3 = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                else:
                    p3 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(p3, f"Mask (thr={args.polygon_threshold})", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Panel 4: polygon overlay
                p4 = frame.copy()
                if polygon_viz is not None:
                    # Scale polygon to frame size if needed
                    poly_h, poly_w = pha_np.shape[:2]
                    scale_x = panel_w / poly_w
                    scale_y = panel_h / poly_h
                    scaled_poly = polygon_viz.copy()
                    scaled_poly[:, 0] *= scale_x
                    scaled_poly[:, 1] *= scale_y
                    p4 = draw_polygon_on_image(p4, scaled_poly, color=(0, 255, 0), thickness=2)
                    p4 = draw_polygon_on_image(p4, scaled_poly, color=(0, 255, 0), thickness=-1)
                    vert_text = f"{len(polygon_viz)} vertices"
                else:
                    vert_text = "No polygon"
                cv2.putText(p4, f"Polygon: {vert_text}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Stack panels: top row [original | alpha], bottom row [mask | polygon]
                top_row = np.hstack([p1, p2])
                bottom_row = np.hstack([p3, p4])
                combined = np.vstack([top_row, bottom_row])

                viz_ms = (time.time() - viz_start) * 1000
                viz_timing.add_timing("visualization", viz_ms)

                # ---- FPS -------------------------------------------------------
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
                last_time = now

                total_ms = (now - loop_start) * 1000

                # ---- status bar on combined image ------------------------------
                result_label = "POLYGON" if result is not None else "NULL"
                verts = len(result["polygon"]) if result else 0
                status = (
                    f"FPS: {fps:.1f} | Total: {total_ms:.0f}ms | "
                    f"Result: {result_label} ({verts} verts) | "
                    f"Alpha: min={pha_np.min():.3f} max={pha_np.max():.3f} mean={pha_np.mean():.3f} | "
                    f"Threshold: {args.polygon_threshold}"
                )
                cv2.putText(combined, status, (10, combined.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Update global frame and stats
                with frame_lock:
                    latest_combined_frame = combined.copy()
                    frame_idx_global = frame_idx
                    avgs = session.timing.get_average_timings()
                    latest_stats = {
                        'fps': fps,
                        'frame': frame_idx,
                        'total_ms': total_ms,
                        'result': result_label,
                        'vertices': verts,
                        'alpha_min': float(pha_np.min()),
                        'alpha_max': float(pha_np.max()),
                        'alpha_mean': float(pha_np.mean()),
                        'decode_ms': avgs.get('data_prep', 0),
                        'infer_ms': avgs.get('model_inference', 0),
                        'poly_ms': avgs.get('generate_polygon', 0),
                        'threshold': args.polygon_threshold,
                    }

                # ---- print to console every 30 frames -------------------------
                frame_idx += 1
                if frame_idx % 30 == 0:
                    avgs = session.timing.get_average_timings()
                    print(
                        f"[frame {frame_idx:5d}] FPS={fps:.1f} "
                        f"decode={avgs.get('data_prep', 0):.1f}ms "
                        f"infer={avgs.get('model_inference', 0):.1f}ms "
                        f"poly={avgs.get('generate_polygon', 0):.1f}ms "
                        f"total={avgs.get('total_frame', 0):.1f}ms "
                        f"| alpha min/max/mean={pha_np.min():.3f}/{pha_np.max():.3f}/{pha_np.mean():.3f} "
                        f"| result={'%d verts' % verts if result else 'None'}"
                    )

                # Save frame if save_dir is set
                if args.save_dir:
                    cv2.imwrite(
                        os.path.join(args.save_dir, f"debug_{frame_idx:06d}.jpg"),
                        combined,
                    )
        except Exception as e:
            print(f"[debug_seg] Error in camera processing loop: {e}")
            import traceback
            traceback.print_exc()

    # Start camera processing in background thread
    processing_thread = threading.Thread(target=camera_processing_loop, daemon=True)
    processing_thread.start()

    print(f"\n[debug_seg] Web server starting...")
    print(f"[debug_seg] Open your browser and go to: http://{args.host}:{args.port}")
    print(f"[debug_seg] Press Ctrl+C to stop\n")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[debug_seg] Interrupted by user")
    finally:
        cap.release()
        print(f"[debug_seg] Done — processed {frame_idx} frames")
        while True:
            loop_start = time.time()

            # #region agent log
            with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"runId":"frame_loop","hypothesisId":"C","location":"debug_segmentation.py:145","message":"Attempting frame read","data":{"frame_idx":frame_idx},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            ret, frame = cap.read()
            # #region agent log
            with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"runId":"frame_loop","hypothesisId":"C","location":"debug_segmentation.py:148","message":"Frame read result","data":{"read_success":ret,"frame_shape":list(frame.shape) if ret and frame is not None else None},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            if not ret:
                print("[debug_seg] Camera read failed, exiting")
                # #region agent log
                with open('/home/jonathan/segment_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"runId":"frame_loop","hypothesisId":"C","location":"debug_segmentation.py:152","message":"Frame read FAILED - exiting loop","data":{"frame_idx":frame_idx},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                break

            # ---- JPEG encode (mimics what the browser does) ----------------
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            ok, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                print("[debug_seg] JPEG encode failed, skipping frame")
                continue
            jpeg_bytes = jpeg_buf.tobytes()

            # ---- run the production process_frame --------------------------
            result = session.process_frame(jpeg_bytes)

            # ---- also run matte_to_polygon with return_mask for viz --------
            # We need the raw alpha matte, so replicate the decode+inference
            # from the session internals.  This is a second inference — we
            # accept the cost for debugging.  To avoid it, we expose the
            # alpha from the session directly:
            viz_start = time.time()

            # Re-decode the same JPEG for the visualization path
            buf_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame_for_viz = cv2.imdecode(buf_np, cv2.IMREAD_COLOR)

            # Run model again to get the alpha matte (for visualisation)
            src_t = to_torch_image(frame_for_viz, device, args.fp16)
            with torch.inference_mode():
                _fgr, pha_t, *_ = model(src_t, *session.rec, args.dsr)
            pha_np = pha_t[0, 0].cpu().numpy()  # float32 [0,1]

            # Get polygon + thresholded mask
            poly_result = matte_to_polygon(
                pha_np,
                threshold=args.polygon_threshold,
                min_area=args.polygon_min_area,
                epsilon_ratio=args.polygon_epsilon,
                return_mask=True,
            )
            if isinstance(poly_result, tuple):
                polygon_viz, mask_viz = poly_result
            else:
                polygon_viz = poly_result
                mask_viz = None

            # ---- build the 4-panel display ---------------------------------
            h, w = frame.shape[:2]
            panel_h, panel_w = h, w

            # Panel 1: original frame
            p1 = frame.copy()
            cv2.putText(p1, "Original", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Panel 2: alpha matte heatmap
            pha_u8 = (pha_np * 255).astype(np.uint8)
            pha_resized = cv2.resize(pha_u8, (panel_w, panel_h))
            p2 = cv2.applyColorMap(pha_resized, cv2.COLORMAP_JET)
            stats_text = f"min={pha_np.min():.3f} max={pha_np.max():.3f} mean={pha_np.mean():.3f}"
            cv2.putText(p2, "Alpha Matte", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(p2, stats_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Panel 3: thresholded mask
            if mask_viz is not None:
                mask_resized = cv2.resize(mask_viz, (panel_w, panel_h))
                p3 = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            else:
                p3 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(p3, f"Mask (thr={args.polygon_threshold})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Panel 4: polygon overlay
            p4 = frame.copy()
            if polygon_viz is not None:
                # Scale polygon to frame size if needed
                poly_h, poly_w = pha_np.shape[:2]
                scale_x = panel_w / poly_w
                scale_y = panel_h / poly_h
                scaled_poly = polygon_viz.copy()
                scaled_poly[:, 0] *= scale_x
                scaled_poly[:, 1] *= scale_y
                p4 = draw_polygon_on_image(p4, scaled_poly, color=(0, 255, 0), thickness=2)
                p4 = draw_polygon_on_image(p4, scaled_poly, color=(0, 255, 0), thickness=-1)
                vert_text = f"{len(polygon_viz)} vertices"
            else:
                vert_text = "No polygon"
            cv2.putText(p4, f"Polygon: {vert_text}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Stack panels: top row [original | alpha], bottom row [mask | polygon]
            top_row = np.hstack([p1, p2])
            bottom_row = np.hstack([p3, p4])
            combined = np.vstack([top_row, bottom_row])

            viz_ms = (time.time() - viz_start) * 1000
            viz_timing.add_timing("visualization", viz_ms)

            # ---- FPS -------------------------------------------------------
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now

            total_ms = (now - loop_start) * 1000

            # ---- status bar on combined image ------------------------------
            result_label = "POLYGON" if result is not None else "NULL"
            verts = len(result["polygon"]) if result else 0
            status = (
                f"FPS: {fps:.1f} | Total: {total_ms:.0f}ms | "
                f"Result: {result_label} ({verts} verts) | "
                f"Alpha: min={pha_np.min():.3f} max={pha_np.max():.3f} mean={pha_np.mean():.3f} | "
                f"Threshold: {args.polygon_threshold}"
            )
            cv2.putText(combined, status, (10, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---- print to console every 30 frames -------------------------
            frame_idx += 1
            if frame_idx % 30 == 0:
                avgs = session.timing.get_average_timings()
                print(
                    f"[frame {frame_idx:5d}] FPS={fps:.1f} "
                    f"decode={avgs.get('data_prep', 0):.1f}ms "
                    f"infer={avgs.get('model_inference', 0):.1f}ms "
                    f"poly={avgs.get('generate_polygon', 0):.1f}ms "
                    f"total={avgs.get('total_frame', 0):.1f}ms "
                    f"| alpha min/max/mean={pha_np.min():.3f}/{pha_np.max():.3f}/{pha_np.mean():.3f} "
                    f"| result={'%d verts' % verts if result else 'None'}"
                )



if __name__ == "__main__":
    main()
