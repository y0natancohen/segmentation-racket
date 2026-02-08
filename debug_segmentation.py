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
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string

from segmentation_server import (
    SegmentationSession,
    load_model,
    to_torch_image,
)
from segmentation.segmentation import (
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


# ---------------------------------------------------------------------------
# Shared state between camera thread and Flask
# ---------------------------------------------------------------------------
latest_combined_frame = None
latest_stats = {}
frame_lock = threading.Lock()
frame_idx_global = 0

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
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { text-align: center; color: #4CAF50; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }
        .video-container {
            text-align: center; margin: 20px 0;
            background: #000; padding: 10px; border-radius: 8px;
        }
        img { max-width: 100%; height: auto; border: 2px solid #4CAF50; border-radius: 4px; }
        .stats {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 20px 0; padding: 15px;
            background: #2a2a2a; border-radius: 8px;
        }
        .stat-item { text-align: center; }
        .stat-label { color: #888; font-size: 12px; margin-bottom: 5px; }
        .stat-value { color: #4CAF50; font-weight: bold; font-size: 18px; }
        .controls { text-align: center; margin: 20px 0; }
        button {
            background: #4CAF50; color: white; border: none;
            padding: 10px 20px; margin: 5px; border-radius: 4px;
            cursor: pointer; font-size: 16px;
        }
        button:hover { background: #45a049; }
        .info { text-align: center; color: #888; margin-top: 20px; font-size: 14px; }
        .panel-labels {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 10px; margin-bottom: 10px; font-size: 12px; color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Debug Segmentation</h1>
        <div class="subtitle">Four-panel view: Original | Alpha Matte | Mask | Polygon Overlay</div>

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
            <div class="stat-item"><div class="stat-label">FPS</div><div class="stat-value" id="fps">0.0</div></div>
            <div class="stat-item"><div class="stat-label">Frame</div><div class="stat-value" id="frame">0</div></div>
            <div class="stat-item"><div class="stat-label">Total Time</div><div class="stat-value" id="total_ms">0ms</div></div>
            <div class="stat-item"><div class="stat-label">Result</div><div class="stat-value" id="result">-</div></div>
            <div class="stat-item"><div class="stat-label">Vertices</div><div class="stat-value" id="vertices">0</div></div>
            <div class="stat-item"><div class="stat-label">Alpha Min</div><div class="stat-value" id="alpha_min">0.000</div></div>
            <div class="stat-item"><div class="stat-label">Alpha Max</div><div class="stat-value" id="alpha_max">0.000</div></div>
            <div class="stat-item"><div class="stat-label">Alpha Mean</div><div class="stat-value" id="alpha_mean">0.000</div></div>
            <div class="stat-item"><div class="stat-label">Decode</div><div class="stat-value" id="decode_ms">0.0ms</div></div>
            <div class="stat-item"><div class="stat-label">Inference</div><div class="stat-value" id="infer_ms">0.0ms</div></div>
            <div class="stat-item"><div class="stat-label">Polygon</div><div class="stat-value" id="poly_ms">0.0ms</div></div>
            <div class="stat-item"><div class="stat-label">Threshold</div><div class="stat-value" id="threshold">0.5</div></div>
        </div>

        <div class="controls">
            <button onclick="saveFrame()">Save Frame</button>
            <button onclick="location.reload()">Refresh</button>
        </div>

        <div class="info">
            Debug segmentation streaming at {{ host }}:{{ port }}<br>
            Press Ctrl+C in terminal to stop
        </div>
    </div>

    <script>
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
                    if (data.success) alert('Frame saved: ' + data.filename);
                    else alert('Error: ' + data.error);
                })
                .catch(err => alert('Error saving frame: ' + err));
        }
    </script>
</body>
</html>
"""


def generate_frames():
    """Generator function for MJPEG video streaming."""
    while True:
        with frame_lock:
            if latest_combined_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_camera(cam_index: int, width: int, height: int, max_retries: int = 3):
    """Open a camera with retries and multiple backends. Returns cap or exits."""
    device_path = f"/dev/video{cam_index}"

    # Permission check
    if os.path.exists(device_path) and not os.access(device_path, os.R_OK):
        print(f"[debug_seg] Cannot read {device_path} — permission denied")
        print(f"[HINT] sudo chmod 666 {device_path}  OR  sudo usermod -a -G video $USER")

    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    cap = None

    for retry in range(max_retries):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(cam_index, backend)
                if cap.isOpened():
                    test_ret, _ = cap.read()
                    if test_ret:
                        break
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
            time.sleep(0.5)
            print(f"[debug_seg] Camera open failed, retrying ({retry + 1}/{max_retries})...")

    if not cap or not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {cam_index} after {max_retries} attempts")
        print(f"[HINT] Make sure the camera is not in use by another application")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[debug_seg] Camera opened: {actual_w}x{actual_h}")

    # Some cameras report 0x0 until first read
    if actual_w == 0 or actual_h == 0:
        test_ret, test_frame = cap.read()
        if test_ret and test_frame is not None:
            actual_w, actual_h = test_frame.shape[1], test_frame.shape[0]
            print(f"[debug_seg] Actual dimensions from test read: {actual_w}x{actual_h}")
        else:
            print("[ERROR] Camera reports 0x0 and test read failed")
            cap.release()
            sys.exit(1)

    return cap


# ---------------------------------------------------------------------------
# Four-panel builder
# ---------------------------------------------------------------------------

def build_debug_panels(frame, pha_np, mask_viz, polygon_viz, threshold):
    """Build the 2x2 debug panel image from raw components."""
    h, w = frame.shape[:2]

    # Panel 1: original frame
    p1 = frame.copy()
    cv2.putText(p1, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 2: alpha matte heatmap
    pha_u8 = (pha_np * 255).astype(np.uint8)
    pha_resized = cv2.resize(pha_u8, (w, h))
    p2 = cv2.applyColorMap(pha_resized, cv2.COLORMAP_JET)
    stats_text = f"min={pha_np.min():.3f} max={pha_np.max():.3f} mean={pha_np.mean():.3f}"
    cv2.putText(p2, "Alpha Matte", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(p2, stats_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 3: thresholded mask
    if mask_viz is not None:
        mask_resized = cv2.resize(mask_viz, (w, h))
        p3 = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    else:
        p3 = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(p3, f"Mask (thr={threshold})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 4: polygon overlay
    p4 = frame.copy()
    if polygon_viz is not None:
        poly_h, poly_w = pha_np.shape[:2]
        scale_x = w / poly_w
        scale_y = h / poly_h
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

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    return np.vstack([top_row, bottom_row])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- device selection ---------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[debug_seg] Device: {device}")

    # ---- load model & session -----------------------------------------------
    model = load_model(args.model_path, device, args.fp16)
    session = SegmentationSession(
        model=model, device=device, fp16=args.fp16, dsr=args.dsr,
        polygon_threshold=args.polygon_threshold,
        polygon_min_area=args.polygon_min_area,
        polygon_epsilon=args.polygon_epsilon,
    )

    # ---- open camera --------------------------------------------------------
    cap = open_camera(args.cam, args.width, args.height)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    viz_timing = TimingStats(max_samples=60)

    # ---- Flask app ----------------------------------------------------------
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, host=args.host, port=args.port)

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/stats')
    def stats():
        with frame_lock:
            return latest_stats.copy()

    @app.route('/save_frame', methods=['POST'])
    def save_frame():
        global latest_combined_frame
        with frame_lock:
            if latest_combined_frame is not None:
                if args.save_dir:
                    filename = os.path.join(args.save_dir, f"debug_{frame_idx_global:06d}.jpg")
                else:
                    filename = f"debug_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, latest_combined_frame)
                return {'success': True, 'filename': filename}
        return {'success': False, 'error': 'No frame available'}, 400

    # ---- camera processing thread -------------------------------------------
    def camera_processing_loop():
        global latest_combined_frame, latest_stats, frame_idx_global

        frame_idx = 0
        fps = 0.0
        last_time = time.time()

        try:
            while True:
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("[debug_seg] Camera read failed, exiting")
                    break

                # JPEG encode (mimics what the browser does)
                ok, jpeg_buf = cv2.imencode(".jpg", frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    print("[debug_seg] JPEG encode failed, skipping frame")
                    continue
                jpeg_bytes = jpeg_buf.tobytes()

                # Run the production process_frame
                result = session.process_frame(jpeg_bytes)

                # Run a second inference for the visualization panels
                viz_start = time.time()
                buf_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame_for_viz = cv2.imdecode(buf_np, cv2.IMREAD_COLOR)

                src_t = to_torch_image(frame_for_viz, device, args.fp16)
                with torch.inference_mode():
                    _fgr, pha_t, *_ = model(src_t, *session.rec, args.dsr)
                pha_np = pha_t[0, 0].cpu().numpy()

                # Polygon + thresholded mask for visualization
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
                    polygon_viz, mask_viz = poly_result, None

                # Build 4-panel image
                combined = build_debug_panels(
                    frame, pha_np, mask_viz, polygon_viz, args.polygon_threshold,
                )
                viz_ms = (time.time() - viz_start) * 1000
                viz_timing.add_timing("visualization", viz_ms)

                # FPS
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
                last_time = now
                total_ms = (now - loop_start) * 1000

                # Status bar
                result_label = "POLYGON" if result is not None else "NULL"
                verts = len(result["polygon"]) if result else 0
                status = (
                    f"FPS: {fps:.1f} | Total: {total_ms:.0f}ms | "
                    f"Result: {result_label} ({verts} verts) | "
                    f"Alpha: min={pha_np.min():.3f} max={pha_np.max():.3f} "
                    f"mean={pha_np.mean():.3f} | Threshold: {args.polygon_threshold}"
                )
                cv2.putText(combined, status, (10, combined.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Update shared state
                with frame_lock:
                    latest_combined_frame = combined.copy()
                    frame_idx_global = frame_idx
                    avgs = session.timing.get_average_timings()
                    latest_stats.update({
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
                    })

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

                if args.save_dir:
                    cv2.imwrite(
                        os.path.join(args.save_dir, f"debug_{frame_idx:06d}.jpg"),
                        combined,
                    )
        except Exception as e:
            print(f"[debug_seg] Error in camera processing loop: {e}")
            import traceback
            traceback.print_exc()

    # ---- start ---------------------------------------------------------------
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
        print(f"[debug_seg] Done — processed {frame_idx_global} frames")


if __name__ == "__main__":
    main()
