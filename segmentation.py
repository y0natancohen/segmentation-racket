import os
import sys
import time
import argparse
import asyncio
import threading
from collections import deque

import cv2
import torch
import numpy as np
from PIL import Image

sys.path.append('rvm')
from model import MattingNetwork

# Import the polygon bridge
from segmentation_polygon_bridge import SegmentationPolygonBridge, set_bridge, send_segmentation_polygon

# Camera and game dimensions constants
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_ASPECT_RATIO = CAMERA_WIDTH / CAMERA_HEIGHT  # 16:9 ≈ 1.78

# Game dimensions (should match camera aspect ratio)
GAME_WIDTH = 800  # 16:9 aspect ratio
GAME_HEIGHT = 450  # 16:9 aspect ratio
GAME_ASPECT_RATIO = GAME_WIDTH / GAME_HEIGHT  # Should equal CAMERA_ASPECT_RATIO


class TimingStats:
    """Track timing statistics for different processing steps"""
    def __init__(self, max_samples=100):
        self.max_samples = max_samples
        self.timings = {
            'camera_capture': deque(maxlen=max_samples),
            'data_prep': deque(maxlen=max_samples),
            'model_inference': deque(maxlen=max_samples),
            'thresholding': deque(maxlen=max_samples),
            'find_contour': deque(maxlen=max_samples),
            'generate_polygon': deque(maxlen=max_samples),
            'build_display': deque(maxlen=max_samples),
            'save_operations': deque(maxlen=max_samples),
            'web_display_update': deque(maxlen=max_samples),
            'fps_calculation': deque(maxlen=max_samples),
            'total_frame': deque(maxlen=max_samples)
        }
    
    def add_timing(self, step_name, duration_ms):
        """Add a timing measurement for a specific step"""
        if step_name in self.timings:
            self.timings[step_name].append(duration_ms)
    
    def get_average_timings(self):
        """Get average timing for each step in milliseconds"""
        averages = {}
        for step, times in self.timings.items():
            if times:
                averages[step] = sum(times) / len(times)
            else:
                averages[step] = 0.0
        return averages
    
    def get_stats_summary(self):
        """Get formatted timing statistics summary"""
        averages = self.get_average_timings()
        summary = []
        for step, avg_time in averages.items():
            if step != 'total_frame':  # Don't include total in individual steps
                summary.append(f"{step.replace('_', ' ').title()}: {avg_time:.1f}ms")
        return summary


def parse_args():
    p = argparse.ArgumentParser(description="Real-time Segmentation Demo", 
                               formatter_class=argparse.RawDescriptionHelpFormatter,
                               epilog="""
Examples:
  # Web display with alpha channel
  python segmentation.py --web_display --show_alpha
  
  # Save original images and segmentation maps
  python segmentation.py --save_images --save_segmentation --output_dir my_output
  
  # Save composite images only
  python segmentation.py --save_composite --headless
  
  # Save polygon images with web display
  python segmentation.py --save_polygon --show_polygon --web_display
  
  # Show thresholded segmentation map
  python segmentation.py --show_threshold --show_alpha --web_display
  
  # Web display with custom background
  python segmentation.py --web_display --bg solid --solid_bgr 255 0 0
                               """)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int, default=CAMERA_WIDTH)
    p.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    p.add_argument("--dsr", type=float, default=0.25, help="downsample_ratio")
    p.add_argument("--bg", choices=["blur","solid","image","transparent"], default="blur")
    p.add_argument("--bg_image", type=str, default=None)
    p.add_argument("--solid_bgr", type=int, nargs=3, default=[60,255,100])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--show_alpha", action="store_true")
    p.add_argument("--headless", action="store_true", help="Run without GUI display")
    p.add_argument("--output_dir", type=str, default="output", help="Output directory for saved frames")
    p.add_argument("--save_images", action="store_true", help="Save original images")
    p.add_argument("--save_segmentation", action="store_true", help="Save segmentation maps")
    p.add_argument("--save_composite", action="store_true", help="Save composite images")
    p.add_argument("--save_polygon", action="store_true", help="Save polygon images")
    p.add_argument("--show_polygon", action="store_true", help="Display polygon overlay on the view")
    p.add_argument("--show_threshold", action="store_true", help="Display thresholded segmentation map")
    p.add_argument("--polygon_threshold", type=float, default=0.5, help="Threshold for polygon extraction")
    p.add_argument("--polygon_min_area", type=int, default=2000, help="Minimum area for polygon extraction")
    p.add_argument("--polygon_epsilon", type=float, default=0.0015, help="Epsilon ratio for polygon simplification")
    p.add_argument("--web_display", action="store_true", help="Display in web browser using simple HTTP server")
    p.add_argument("--web_port", type=int, default=8080, help="Port for web display")
    p.add_argument("--polygon_bridge", action="store_true", help="Send polygon data to Phaser game via WebSocket")
    p.add_argument("--polygon_bridge_port", type=int, default=8765, help="Port for polygon bridge WebSocket")
    return p.parse_args()

def to_torch_image(frame_bgr, device, half):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).to(device).permute(2,0,1).float()/255.0
    if half and device.type == "cuda": ten = ten.half()
    return ten.unsqueeze(0)

def to_torch_bg(image_bgr, device, half):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).to(device).permute(2,0,1).float()/255.0
    if half and device.type == "cuda": ten = ten.half()
    return ten


def setup_web_display(port):
    """Setup simple HTTP server for web display"""
    import http.server
    import socketserver
    import threading
    import json
    from urllib.parse import urlparse, parse_qs
    
    # Global variables to store the latest frame and system info
    latest_frame_data = [None]
    system_info = [{"fps": 0.0, "method": "RVM", "dsr": 0.25, "fp16": False}]
    
    class WebDisplayHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Real-time Segmentation</title>
                    <style>
                        body { margin: 0; padding: 20px; background: #000; color: #fff; font-family: Arial; }
                        .container { max-width: 1200px; margin: 0 auto; }
                        .frame { margin: 10px 0; border: 2px solid #333; }
                        .frame img { width: 100%; height: auto; }
                        .status { background: #333; padding: 10px; margin: 10px 0; border-radius: 5px; }
                        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }
                        .info-card { background: #222; padding: 10px; border-radius: 5px; border-left: 4px solid #00ff00; }
                        .info-label { font-size: 12px; color: #888; }
                        .info-value { font-size: 18px; font-weight: bold; color: #00ff00; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Real-time Segmentation Demo</h1>
                        <div class="info-grid">
                            <div class="info-card">
                                <div class="info-label">FPS</div>
                                <div class="info-value" id="fps">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Method</div>
                                <div class="info-value" id="method">RVM</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Downsample Ratio</div>
                                <div class="info-value" id="dsr">0.25</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">FP16</div>
                                <div class="info-value" id="fp16">False</div>
                            </div>
                        </div>
                        <h2>Timing Statistics (ms)</h2>
                        <div class="info-grid">
                            <div class="info-card">
                                <div class="info-label">Camera Capture</div>
                                <div class="info-value" id="timing-camera_capture">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Data Prep</div>
                                <div class="info-value" id="timing-data_prep">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Model Inference</div>
                                <div class="info-value" id="timing-model_inference">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Thresholding</div>
                                <div class="info-value" id="timing-thresholding">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Find Contour</div>
                                <div class="info-value" id="timing-find_contour">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Generate Polygon</div>
                                <div class="info-value" id="timing-generate_polygon">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Build Display</div>
                                <div class="info-value" id="timing-build_display">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Save Operations</div>
                                <div class="info-value" id="timing-save_operations">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Web Display</div>
                                <div class="info-value" id="timing-web_display_update">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">FPS Calculation</div>
                                <div class="info-value" id="timing-fps_calculation">0.0</div>
                            </div>
                            <div class="info-card">
                                <div class="info-label">Total Frame</div>
                                <div class="info-value" id="timing-total_frame">0.0</div>
                            </div>
                        </div>
                        <div class="frame">
                            <img id="frame" src="/frame" alt="Live Feed">
                        </div>
                    </div>
                    <script>
                        function updateFrame() {
                            document.getElementById('frame').src = '/frame?' + new Date().getTime();
                        }
                        function updateInfo() {
                            fetch('/info')
                                .then(response => response.json())
                                .then(data => {
                                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                                    document.getElementById('method').textContent = data.method;
                                    document.getElementById('dsr').textContent = data.dsr;
                                    document.getElementById('fp16').textContent = data.fp16 ? 'True' : 'False';
                                    
                                    // Update timing statistics
                                    if (data.timing) {
                                        for (const [key, value] of Object.entries(data.timing)) {
                                            const element = document.getElementById('timing-' + key);
                                            if (element) {
                                                element.textContent = value.toFixed(1);
                                            }
                                        }
                                    }
                                })
                                .catch(err => console.log('Info update failed:', err));
                        }
                        setInterval(updateFrame, 100); // Update every 100ms
                        setInterval(updateInfo, 500); // Update info every 500ms
                    </script>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
            elif self.path.startswith('/frame'):
                if latest_frame_data[0] is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(latest_frame_data[0])
                else:
                    self.send_response(404)
                    self.end_headers()
            elif self.path == '/info':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(json.dumps(system_info[0]).encode())
            else:
                self.send_response(404)
                self.end_headers()
    
    # Try to create the server and handle port binding errors gracefully
    try:
        handler = WebDisplayHandler
        httpd = socketserver.TCPServer(("", port), handler)
        print(f"Web display server started at http://localhost:{port}")
        return httpd, handler, latest_frame_data, system_info
    except OSError as e:
        if e.errno == 98:  # Address already in use
            raise RuntimeError(f"Port {port} is already in use. Please stop any existing segmentation processes or use a different port with --web_port")
        else:
            raise RuntimeError(f"Failed to bind to port {port}: {e}")

def setup_model(args):
    """Setup and load the RVM model"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    
    # Create model directory if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Download model weights if not present
    model_path = os.path.join(model_dir, "rvm_mobilenetv3.pth")
    if not os.path.exists(model_path):
        print("Downloading RVM MobileNetV3 model weights...")
        import urllib.request
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
        urllib.request.urlretrieve(url, model_path)
        print("Model weights downloaded successfully!")
    
    # Load the model
    model = MattingNetwork('mobilenetv3').eval().to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev, weights_only=True))
    if args.fp16 and dev.type == "cuda": 
        model = model.half()
    print("RVM model loaded successfully from local submodule!")
    
    return model, dev

def setup_camera(args):
    """Setup camera capture"""
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual FPS: {cam_fps}")
    return cap

def setup_background(args):
    """Setup background image if specified"""
    bg_img = None
    if args.bg == "image":
        if not args.bg_image or not os.path.exists(args.bg_image):
            raise SystemExit("--bg image requires --bg_image")
        bg_img = cv2.imread(args.bg_image, cv2.IMREAD_COLOR)
        if bg_img is None: 
            raise SystemExit(f"Failed to load {args.bg_image}")
    return bg_img

def setup_display(args):
    """Setup web display if requested"""
    web_server = None
    web_handler = None
    latest_frame_data = None
    system_info = None
    
    if args.web_display:
        try:
            web_server, web_handler, latest_frame_data, system_info = setup_web_display(args.web_port)
            import threading
            web_thread = threading.Thread(target=web_server.serve_forever, daemon=True)
            web_thread.start()
        except Exception as e:
            print(f"Failed to start web display: {e}")
            raise e
    
    return web_server, web_handler, latest_frame_data, system_info

def setup_output_directories(args):
    """Create output directories if saving files"""
    if args.save_images or args.save_segmentation or args.save_composite or args.save_polygon or args.headless:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        if args.save_segmentation:
            os.makedirs(os.path.join(args.output_dir, "segmentation"), exist_ok=True)
        if args.save_composite:
            os.makedirs(os.path.join(args.output_dir, "composite"), exist_ok=True)
        if args.save_polygon:
            os.makedirs(os.path.join(args.output_dir, "polygon"), exist_ok=True)

def setup_signal_handlers(web_server, cap):
    """Setup signal handlers for graceful shutdown"""
    import signal
    import sys
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        try:
            if web_server is not None:
                print("Shutting down web server...")
                web_server.shutdown()
                web_server.server_close()
            cap.release()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            print("Shutdown complete")
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def process_frame(frame, model, dev, args, bg_img, rec, timing_stats=None):
    """Process a single frame with RVM model"""
    frame_start = time.time()
    
    # Data preparation timing
    prep_start = time.time()
    H, W = frame.shape[:2]
    
    # Prepare background
    if args.bg == "blur":
        k = max(3, (min(H,W)//50)*2+1)
        bg = cv2.GaussianBlur(frame, (k,k), 0)
    elif args.bg == "solid":
        bg = np.full((H,W,3), args.solid_bgr, dtype=np.uint8)
    elif args.bg == "image":
        bg = cv2.resize(bg_img, (W, H), interpolation=cv2.INTER_AREA)
    else:
        bg = np.zeros((H,W,3), dtype=np.uint8)

    # Use RVM model
    src = to_torch_image(frame, dev, args.fp16)
    bgT = to_torch_bg(bg, dev, args.fp16)
    prep_time = (time.time() - prep_start) * 1000
    if timing_stats:
        timing_stats.add_timing('data_prep', prep_time)

    # Model inference timing
    inference_start = time.time()
    fgr, pha, rec[0], rec[1], rec[2], rec[3] = model(src, rec[0], rec[1], rec[2], rec[3], args.dsr)
    com = fgr * pha + bgT.unsqueeze(0) * (1 - pha)  # from RVM README
    com = (com.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    com_bgr = cv2.cvtColor(com, cv2.COLOR_RGB2BGR)
    
    # Extract segmentation map (alpha channel)
    seg_map = (pha[0,0].cpu().numpy()*255).astype(np.uint8)
    seg_map_bgr = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
    inference_time = (time.time() - inference_start) * 1000
    if timing_stats:
        timing_stats.add_timing('model_inference', inference_time)
    
    # Generate polygon from segmentation map
    polygon = None
    thresholded_mask = None
    if args.save_polygon or args.show_polygon or args.show_threshold:
        # Convert segmentation map to float32 for polygon extraction
        pha_float = pha[0,0].cpu().numpy().astype(np.float32)
        if args.show_threshold:
            polygon, thresholded_mask = matte_to_polygon(
                pha_float, 
                threshold=args.polygon_threshold,
                min_area=args.polygon_min_area,
                epsilon_ratio=args.polygon_epsilon,
                return_mask=True,
                timing_stats=timing_stats
            )
        else:
            polygon = matte_to_polygon(
                pha_float, 
                threshold=args.polygon_threshold,
                min_area=args.polygon_min_area,
                epsilon_ratio=args.polygon_epsilon,
                timing_stats=timing_stats
            )

    total_time = (time.time() - frame_start) * 1000
    if timing_stats:
        timing_stats.add_timing('total_frame', total_time)

    return com_bgr, seg_map_bgr, polygon, thresholded_mask, rec

def build_view(frame, com_bgr, seg_map_bgr, thresholded_mask, args, timing_stats=None):
    """Build the display view with different components"""
    build_start = time.time()
    
    view_components = [frame, com_bgr, seg_map_bgr]
    
    # Add thresholded mask if requested
    if args.show_threshold and thresholded_mask is not None:
        thresholded_bgr = cv2.cvtColor(thresholded_mask, cv2.COLOR_GRAY2BGR)
        view_components.append(thresholded_bgr)
    
    if args.show_alpha or args.show_threshold:
        view = np.hstack(view_components)
    else:
        view = com_bgr
    
    build_time = (time.time() - build_start) * 1000
    if timing_stats:
        timing_stats.add_timing('build_display', build_time)
    
    return view

def save_frame_data(frame, com_bgr, seg_map, polygon, args, frame_count, timing_stats=None):
    """Save frame data if requested"""
    save_start = time.time()
    
    if args.save_images:
        cv2.imwrite(os.path.join(args.output_dir, "images", f"frame_{frame_count:06d}.jpg"), frame)
    
    if args.save_segmentation:
        cv2.imwrite(os.path.join(args.output_dir, "segmentation", f"seg_{frame_count:06d}.jpg"), seg_map)
    
    if args.save_composite:
        cv2.imwrite(os.path.join(args.output_dir, "composite", f"com_{frame_count:06d}.jpg"), com_bgr)
    
    if args.save_polygon and polygon is not None:
        # Save image with polygon overlay
        polygon_image = draw_polygon_on_image(frame, polygon, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(args.output_dir, "polygon", f"poly_{frame_count:06d}.jpg"), polygon_image)
    
    save_time = (time.time() - save_start) * 1000
    if timing_stats:
        timing_stats.add_timing('save_operations', save_time)

def update_web_display(view, latest_frame_data, timing_stats=None):
    """Update web display with current frame"""
    web_start = time.time()
    
    if latest_frame_data is not None:
        _, buffer = cv2.imencode('.jpg', view)
        latest_frame_data[0] = buffer.tobytes()
    
    web_time = (time.time() - web_start) * 1000
    if timing_stats:
        timing_stats.add_timing('web_display_update', web_time)

def update_status_display(view, fps, args, polygon, system_info, timing_stats=None):
    """Update status display with FPS and polygon info"""
    status_text = f"FPS:{fps:5.1f} Method:RVM dsr={args.dsr} fp16={args.fp16}"
    
    # Update system info for web display
    if system_info is not None:
        timing_data = {}
        if timing_stats:
            timing_data = timing_stats.get_average_timings()
        
        system_info[0] = {
            "fps": fps,
            "method": "RVM",
            "dsr": args.dsr,
            "fp16": args.fp16,
            "timing": timing_data
        }
    
    cv2.putText(view, status_text, (12,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(view, status_text, (12,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    
    # Add polygon overlay if requested
    if args.show_polygon and polygon is not None:
        # Draw polygon on the view
        view = draw_polygon_on_image(view, polygon, color=(0, 255, 0), thickness=2)
        
        # Add polygon info to status
        polygon_info = f" Polygon: {len(polygon)} vertices"
        cv2.putText(view, polygon_info, (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(view, polygon_info, (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    # Add threshold info if threshold display is enabled
    if args.show_threshold:
        threshold_info = f" Threshold: {args.polygon_threshold}"
        y_pos = 90 if args.show_polygon and polygon is not None else 60
        cv2.putText(view, threshold_info, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(view, threshold_info, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    return view

def calculate_fps(last_time, current_time):
    """Calculate FPS from time difference"""
    return 1.0 / (current_time - last_time) if current_time > last_time else 0.0

def cleanup_resources(cap, web_server):
    """Clean up all resources"""
    cap.release()
    if web_server is not None:
        print("Shutting down web server...")
        web_server.shutdown()
        web_server.server_close()

def run_segmentation_loop(args, model, dev, cap, bg_img, web_server, latest_frame_data, system_info, polygon_bridge=None):
    """Main segmentation processing loop"""
    rec = [None, None, None, None]
    last = time.time()
    fps = 0.0
    frame_count = 0
    timing_stats = TimingStats()
    
    # Start polygon bridge server if enabled
    bridge_thread = None
    if polygon_bridge:
        def run_bridge():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(polygon_bridge.start_server())
        
        bridge_thread = threading.Thread(target=run_bridge, daemon=True)
        bridge_thread.start()
        time.sleep(1)  # Give the server time to start
    
    with torch.inference_mode():
        while True:
            loop_start = time.time()
            
            # Camera capture timing
            camera_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            camera_time = (time.time() - camera_start) * 1000
            timing_stats.add_timing('camera_capture', camera_time)
            
            # Process frame
            com_bgr, seg_map_bgr, polygon, thresholded_mask, rec = process_frame(
                frame, model, dev, args, bg_img, rec, timing_stats)
            
            # Build view
            view = build_view(frame, com_bgr, seg_map_bgr, thresholded_mask, args, timing_stats)
            
            # FPS calculation timing
            fps_start = time.time()
            now = time.time()
            fps = 0.9 * fps + 0.1 * calculate_fps(last, now)
            last = now
            fps_time = (time.time() - fps_start) * 1000
            timing_stats.add_timing('fps_calculation', fps_time)
            
            # Send polygon data to bridge if enabled
            if polygon_bridge and polygon is not None:
                try:
                    # Send raw polygon coordinates - scaling will be handled by the bridge
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Send polygon data synchronously using the bridge's method
                    # Create a new event loop for this call
                    def send_polygon():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(polygon_bridge.send_polygon_data(
                            polygon,  # Send raw polygon coordinates
                            frame_size=(frame_width, frame_height)
                        ))
                        loop.close()
                    
                    # Run in a separate thread to avoid blocking
                    threading.Thread(target=send_polygon, daemon=True).start()
                except Exception as e:
                    print(f"Error sending polygon to bridge: {e}")
            
            # Update status display
            view = update_status_display(view, fps, args, polygon, system_info, timing_stats)
            
            # Save frame data if requested
            if args.save_images or args.save_segmentation or args.save_composite or args.save_polygon:
                save_frame_data(frame, com_bgr, seg_map_bgr[:,:,0], polygon, args, frame_count, timing_stats)
                frame_count += 1
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    print(f"Processed {frame_count} frames...")

            # Update web display
            if args.web_display:
                update_web_display(view, latest_frame_data, timing_stats)
            
            # Total frame timing
            total_time = (time.time() - loop_start) * 1000
            timing_stats.add_timing('total_frame', total_time)
            
            # Debug: Print timing breakdown every 30 frames
            if frame_count % 30 == 0:
                averages = timing_stats.get_average_timings()
                print(f"Timing breakdown: Camera={averages['camera_capture']:.1f}ms, "
                      f"Processing={averages['data_prep'] + averages['model_inference'] + averages['thresholding'] + averages['find_contour'] + averages['generate_polygon'] + averages['build_display']:.1f}ms, "
                      f"Total={averages['total_frame']:.1f}ms")
    
    # Cleanup bridge thread
    if bridge_thread and bridge_thread.is_alive():
        # The thread will be cleaned up automatically as a daemon thread
        pass

def setup_polygon_bridge(args):
    """Setup polygon bridge for sending data to Phaser game"""
    if args.polygon_bridge:
        bridge = SegmentationPolygonBridge(port=args.polygon_bridge_port)
        set_bridge(bridge)
        return bridge
    return None

def main():
    """Main function - orchestrates setup, runs segmentation loop, and cleanup"""
    args = parse_args()
    
    # Setup components
    model, dev = setup_model(args)
    cap = setup_camera(args)
    bg_img = setup_background(args)
    web_server, web_handler, latest_frame_data, system_info = setup_display(args)
    setup_output_directories(args)
    setup_signal_handlers(web_server, cap)
    polygon_bridge = setup_polygon_bridge(args)

    # Run main segmentation loop
    run_segmentation_loop(args, model, dev, cap, bg_img, web_server, latest_frame_data, system_info, polygon_bridge)

    # Cleanup
    cleanup_resources(cap, web_server)
    if polygon_bridge:
        polygon_bridge.stop()


def matte_to_polygon(pha, threshold=0.5, min_area=2000, epsilon_ratio=0.015, return_mask=False, timing_stats=None):
    """
    pha: 2D float32 array in [0,1] (alpha matte)
    threshold: binarization threshold for foreground
    min_area: ignore tiny contours below this many pixels
    epsilon_ratio: Douglas-Peucker epsilon as a fraction of perimeter
    return_mask: if True, return both polygon and thresholded mask
    timing_stats: TimingStats object to record timing measurements
    Returns: Nx2 float32 array of (x,y) polygon vertices, or None if none found.
             If return_mask=True, returns (polygon, thresholded_mask) tuple.
    """
    H, W = pha.shape[:2]

    # 1) Smooth and threshold
    threshold_start = time.time()
    m = cv2.GaussianBlur(pha, (0,0), 1.2)
    mask = (m >= threshold).astype(np.uint8) * 255
    threshold_time = (time.time() - threshold_start) * 1000
    if timing_stats:
        timing_stats.add_timing('thresholding', threshold_time)

    # 2) Morphology to clean noise & holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 3) Contour extraction
    contour_start = time.time()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        contour_time = (time.time() - contour_start) * 1000
        if timing_stats:
            timing_stats.add_timing('find_contour', contour_time)
        return (None, mask) if return_mask else None
    
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area: 
        contour_time = (time.time() - contour_start) * 1000
        if timing_stats:
            timing_stats.add_timing('find_contour', contour_time)
        return (None, mask) if return_mask else None
    
    # Generate polygon
    polygon_start = time.time()
    peri = cv2.arcLength(cnt, True)
    eps  = epsilon_ratio * peri
    poly = cv2.approxPolyDP(cnt, eps, True)  # Douglas–Peucker
    poly = poly.reshape(-1, 2).astype(np.float32)
    polygon_time = (time.time() - polygon_start) * 1000
    if timing_stats:
        timing_stats.add_timing('generate_polygon', polygon_time)
    
    contour_time = (time.time() - contour_start) * 1000
    if timing_stats:
        timing_stats.add_timing('find_contour', contour_time)
    
    return (poly, mask) if return_mask else poly


def draw_polygon_on_image(image, polygon, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
    """
    Draw polygon on image with optional fill.
    
    Args:
        image: BGR image array
        polygon: Nx2 array of (x,y) vertices
        color: BGR color tuple
        thickness: line thickness (-1 for filled)
        fill_alpha: transparency for fill (0.0 = transparent, 1.0 = opaque)
    
    Returns:
        Image with polygon drawn
    """
    if polygon is None or len(polygon) < 3:
        return image
    
    # Convert to integer coordinates
    pts = polygon.astype(np.int32)
    
    # Create overlay for transparency
    overlay = image.copy()
    
    # Draw filled polygon if thickness is -1
    if thickness == -1:
        cv2.fillPoly(overlay, [pts], color)
        # Blend with original image
        image = cv2.addWeighted(image, 1 - fill_alpha, overlay, fill_alpha, 0)
    else:
        # Draw polygon outline
        cv2.polylines(image, [pts], True, color, thickness)
    
    return image


if __name__ == "__main__":
    main()
