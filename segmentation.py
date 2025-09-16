import os
import sys
import time
import argparse

import cv2
import torch
import numpy as np
from PIL import Image

sys.path.append('rvm')
from model import MattingNetwork


def parse_args():
    p = argparse.ArgumentParser(description="Real-time Segmentation Demo with RVM", 
                               formatter_class=argparse.RawDescriptionHelpFormatter,
                               epilog="""
Examples:
  # Live display with alpha channel
  python segmentation.py --live_display --show_alpha
  
  # Web display (works without GUI support)
  python segmentation.py --web_display --show_alpha
  
  # Save original images and segmentation maps
  python segmentation.py --save_images --save_segmentation --output_dir my_output
  
  # Save composite images only
  python segmentation.py --save_composite --headless
  
  # Save polygon images with display
  python segmentation.py --save_polygon --show_polygon --live_display
  
  # Show thresholded segmentation map
  python segmentation.py --show_threshold --show_alpha --live_display
  
  # Live display with custom background
  python segmentation.py --live_display --bg solid --solid_bgr 255 0 0
                               """)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
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
    p.add_argument("--live_display", action="store_true", help="Display live on screen (overrides headless)")
    p.add_argument("--web_display", action="store_true", help="Display in web browser using simple HTTP server")
    p.add_argument("--web_port", type=int, default=8080, help="Port for web display")
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

def simple_background_removal(frame, bg):
    """Simple background removal using color-based segmentation"""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color (adjust these values as needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Convert mask to 3-channel
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    # Apply the mask
    result = frame * mask_3ch + bg * (1 - mask_3ch)
    
    return result.astype(np.uint8), mask

def check_port_available(port):
    """Check if a port is available, raise error if not"""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except OSError as e:
        if e.errno == 98:  # Address already in use
            raise RuntimeError(f"Port {port} is already in use. Please stop any existing segmentation processes or use a different port with --web_port")
        else:
            raise RuntimeError(f"Failed to bind to port {port}: {e}")

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
    
    # Check if port is available
    check_port_available(port)
    
    handler = WebDisplayHandler
    httpd = socketserver.TCPServer(("", port), handler)
    print(f"Web display server started at http://localhost:{port}")
    return httpd, handler, latest_frame_data, system_info

def main():
    a = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    # Load RVM (MobileNetV3 = fast) - Using local submodule
    try:
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
        if a.fp16 and dev.type == "cuda": 
            model = model.half()
        use_rvm = True
        print("RVM model loaded successfully from local submodule!")
    except Exception as e:
        print(f"Failed to load RVM model: {e}")
        print("Using OpenCV-based background removal as fallback...")
        use_rvm = False

    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    bg_img = None
    if a.bg == "image":
        if not a.bg_image or not os.path.exists(a.bg_image):
            raise SystemExit("--bg image requires --bg_image")
        bg_img = cv2.imread(a.bg_image, cv2.IMREAD_COLOR)
        if bg_img is None: raise SystemExit(f"Failed to load {a.bg_image}")

    rec = [None, None, None, None]
    last = time.time(); fps = 0.0
    frame_count = 0
    
    # Check GUI support
    gui_supported = True
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("test", test_img)
        cv2.destroyAllWindows()
    except:
        gui_supported = False
        print("GUI display not supported - OpenCV compiled without GUI support")
    
    # Determine display mode
    if a.web_display:
        display_mode = "web"
    elif a.live_display and gui_supported:
        display_mode = "live"
    elif a.headless or not gui_supported:
        display_mode = "headless"
    else:
        display_mode = "live" if gui_supported else "headless"
    
    # Create output directories if saving
    if a.save_images or a.save_segmentation or a.save_composite or a.save_polygon or a.headless:
        os.makedirs(a.output_dir, exist_ok=True)
        if a.save_images:
            os.makedirs(os.path.join(a.output_dir, "images"), exist_ok=True)
        if a.save_segmentation:
            os.makedirs(os.path.join(a.output_dir, "segmentation"), exist_ok=True)
        if a.save_composite:
            os.makedirs(os.path.join(a.output_dir, "composite"), exist_ok=True)
        if a.save_polygon:
            os.makedirs(os.path.join(a.output_dir, "polygon"), exist_ok=True)
        print(f"Output will be saved to: {a.output_dir}")
    
    print(f"Running in {display_mode} mode")
    
    # Setup web display if requested
    web_server = None
    web_handler = None
    latest_frame_data = None
    system_info = None
    if display_mode == "web":
        try:
            web_server, web_handler, latest_frame_data, system_info = setup_web_display(a.web_port)
            import threading
            web_thread = threading.Thread(target=web_server.serve_forever, daemon=True)
            web_thread.start()
        except Exception as e:
            print(f"Failed to start web server: {e}")
            print("Falling back to headless mode")
            display_mode = "headless"

    # Setup signal handling for graceful shutdown
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
            # Only try to destroy windows if GUI is supported
            if gui_supported:
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            print("Shutdown complete")
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok: break
            H, W = frame.shape[:2]

            if a.bg == "blur":
                k = max(3, (min(H,W)//50)*2+1)
                bg = cv2.GaussianBlur(frame, (k,k), 0)
            elif a.bg == "solid":
                bg = np.full((H,W,3), a.solid_bgr, dtype=np.uint8)
            elif a.bg == "image":
                bg = cv2.resize(bg_img, (W, H), interpolation=cv2.INTER_AREA)
            else:
                bg = np.zeros((H,W,3), dtype=np.uint8)

            if use_rvm:
                # Use RVM model
                src = to_torch_image(frame, dev, a.fp16)
                bgT = to_torch_bg(bg, dev, a.fp16)

                fgr, pha, rec[0], rec[1], rec[2], rec[3] = model(src, rec[0], rec[1], rec[2], rec[3], a.dsr)
                com = fgr * pha + bgT.unsqueeze(0) * (1 - pha)  # from RVM README
                com = (com.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                com_bgr = cv2.cvtColor(com, cv2.COLOR_RGB2BGR)
                
                # Extract segmentation map (alpha channel)
                seg_map = (pha[0,0].cpu().numpy()*255).astype(np.uint8)
                seg_map_bgr = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
                
                # Generate polygon from segmentation map
                polygon = None
                thresholded_mask = None
                if a.save_polygon or a.show_polygon or a.show_threshold:
                    # Convert segmentation map to float32 for polygon extraction
                    pha_float = pha[0,0].cpu().numpy().astype(np.float32)
                    if a.show_threshold:
                        polygon, thresholded_mask = matte_to_polygon(
                            pha_float, 
                            threshold=a.polygon_threshold,
                            min_area=a.polygon_min_area,
                            epsilon_ratio=a.polygon_epsilon,
                            return_mask=True
                        )
                    else:
                        polygon = matte_to_polygon(
                            pha_float, 
                            threshold=a.polygon_threshold,
                            min_area=a.polygon_min_area,
                            epsilon_ratio=a.polygon_epsilon
                        )

                # Build view with different components
                view_components = [frame, com_bgr, seg_map_bgr]
                
                # Add thresholded mask if requested
                if a.show_threshold and thresholded_mask is not None:
                    thresholded_bgr = cv2.cvtColor(thresholded_mask, cv2.COLOR_GRAY2BGR)
                    view_components.append(thresholded_bgr)
                
                if a.show_alpha or a.show_threshold:
                    view = np.hstack(view_components)
                else:
                    view = com_bgr
            else:
                # Use simple OpenCV-based background removal
                com_bgr, mask = simple_background_removal(frame, bg)
                seg_map = mask
                seg_map_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Generate polygon from segmentation map
                polygon = None
                thresholded_mask = None
                if a.save_polygon or a.show_polygon or a.show_threshold:
                    # Convert mask to float32 for polygon extraction
                    mask_float = mask.astype(np.float32) / 255.0
                    if a.show_threshold:
                        polygon, thresholded_mask = matte_to_polygon(
                            mask_float, 
                            threshold=a.polygon_threshold,
                            min_area=a.polygon_min_area,
                            epsilon_ratio=a.polygon_epsilon,
                            return_mask=True
                        )
                    else:
                        polygon = matte_to_polygon(
                            mask_float, 
                            threshold=a.polygon_threshold,
                            min_area=a.polygon_min_area,
                            epsilon_ratio=a.polygon_epsilon
                        )
                
                # Build view with different components
                view_components = [frame, com_bgr, seg_map_bgr]
                
                # Add thresholded mask if requested
                if a.show_threshold and thresholded_mask is not None:
                    thresholded_bgr = cv2.cvtColor(thresholded_mask, cv2.COLOR_GRAY2BGR)
                    view_components.append(thresholded_bgr)
                
                if a.show_alpha or a.show_threshold:
                    view = np.hstack(view_components)
                else:
                    view = com_bgr

            now = time.time(); fps = 0.9*fps + 0.1*(1.0/(now-last)); last = now
            method_text = "RVM" if use_rvm else "OpenCV"
            status_text = f"FPS:{fps:5.1f} Method:{method_text}"
            if use_rvm:
                status_text += f" dsr={a.dsr} fp16={a.fp16}"
            
            # Update system info for web display
            if system_info is not None:
                system_info[0] = {
                    "fps": fps,
                    "method": method_text,
                    "dsr": a.dsr,
                    "fp16": a.fp16
                }
            
            cv2.putText(view, status_text, (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(view, status_text, (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            
            # Add polygon overlay if requested
            if a.show_polygon and polygon is not None:
                # Draw polygon on the view
                view = draw_polygon_on_image(view, polygon, color=(0, 255, 0), thickness=2)
                
                # Add polygon info to status
                polygon_info = f" Polygon: {len(polygon)} vertices"
                cv2.putText(view, polygon_info, (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(view, polygon_info, (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            # Add threshold info if threshold display is enabled
            if a.show_threshold:
                threshold_info = f" Threshold: {a.polygon_threshold}"
                y_pos = 90 if a.show_polygon and polygon is not None else 60
                cv2.putText(view, threshold_info, (12, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(view, threshold_info, (12, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # Save individual components if requested
            if a.save_images or a.save_segmentation or a.save_composite or a.save_polygon or a.headless:
                frame_id = f"{frame_count:06d}"
                
                if a.save_images:
                    img_path = os.path.join(a.output_dir, "images", f"image_{frame_id}.jpg")
                    cv2.imwrite(img_path, frame)
                
                if a.save_segmentation:
                    seg_path = os.path.join(a.output_dir, "segmentation", f"seg_{frame_id}.jpg")
                    cv2.imwrite(seg_path, seg_map)
                
                if a.save_composite:
                    comp_path = os.path.join(a.output_dir, "composite", f"comp_{frame_id}.jpg")
                    cv2.imwrite(comp_path, com_bgr)
                
                if a.save_polygon and polygon is not None:
                    # Create polygon visualization image
                    poly_img = frame.copy()
                    poly_img = draw_polygon_on_image(poly_img, polygon, color=(0, 255, 0), thickness=3)
                    
                    # Add polygon info text
                    poly_info = f"Vertices: {len(polygon)}"
                    cv2.putText(poly_img, poly_info, (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(poly_img, poly_info, (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
                    
                    poly_path = os.path.join(a.output_dir, "polygon", f"poly_{frame_id}.jpg")
                    cv2.imwrite(poly_path, poly_img)
                
                # Save combined view in headless mode (legacy behavior)
                if a.headless and not (a.save_images or a.save_segmentation or a.save_composite):
                    output_path = os.path.join(a.output_dir, f"frame_{frame_id}.jpg")
                    cv2.imwrite(output_path, view)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    print(f"Processed {frame_count} frames...")

            # Display based on mode
            if display_mode == "web":
                # Send frame to web display
                if latest_frame_data is not None:
                    _, buffer = cv2.imencode('.jpg', view)
                    latest_frame_data[0] = buffer.tobytes()
            elif display_mode == "live":
                window_title = "Real-time Segmentation Demo - q to quit"
                try:
                    cv2.imshow(window_title, view)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                except cv2.error as e:
                    print(f"GUI display error: {e}")
                    print("Switching to headless mode...")
                    display_mode = "headless"
                    if not (a.save_images or a.save_segmentation or a.save_composite):
                        os.makedirs(a.output_dir, exist_ok=True)

    # Cleanup
    cap.release()
    # Only try to destroy windows if GUI is supported
    if gui_supported:
        cv2.destroyAllWindows()
    if web_server is not None:
        print("Shutting down web server...")
        web_server.shutdown()
        web_server.server_close()


def matte_to_polygon(pha, threshold=0.5, min_area=2000, epsilon_ratio=0.015, use_skimage=False, return_mask=False):
    """
    pha: 2D float32 array in [0,1] (alpha matte)
    threshold: binarization threshold for foreground
    min_area: ignore tiny contours below this many pixels
    epsilon_ratio: Douglas-Peucker epsilon as a fraction of perimeter
    use_skimage: if True, use skimage.measure.find_contours (subpixel)
    return_mask: if True, return both polygon and thresholded mask
    Returns: Nx2 float32 array of (x,y) polygon vertices, or None if none found.
             If return_mask=True, returns (polygon, thresholded_mask) tuple.
    """
    H, W = pha.shape[:2]

    # 1) Smooth and threshold
    m = cv2.GaussianBlur(pha, (0,0), 1.2)
    mask = (m >= threshold).astype(np.uint8) * 255

    # 2) Morphology to clean noise & holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 3) Contour extraction
    if not use_skimage:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            return (None, mask) if return_mask else None
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area: 
            return (None, mask) if return_mask else None
        peri = cv2.arcLength(cnt, True)
        eps  = epsilon_ratio * peri
        poly = cv2.approxPolyDP(cnt, eps, True)  # Douglas–Peucker
        poly = poly.reshape(-1, 2).astype(np.float32)
        return (poly, mask) if return_mask else poly
    else:
        # Subpixel option via marching squares
        from skimage import measure
        # find_contours returns (row, col) in float coords
        contours = measure.find_contours(mask.astype(np.float32)/255.0, 0.5)
        if not contours: 
            return (None, mask) if return_mask else None
        # choose largest by polygon area (convert to x,y)
        def area_xy(pts):
            x = pts[:,1]; y = pts[:,0]
            return 0.5*np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
        largest = max(contours, key=lambda c: area_xy(c))
        xy = np.stack([largest[:,1], largest[:,0]], axis=1).astype(np.float32)  # (x,y)
        # Optional: simplify with RDP (OpenCV needs integer input; cast then back)
        xy_int = xy.astype(np.int32).reshape(-1,1,2)
        peri = cv2.arcLength(xy_int, True)
        eps  = epsilon_ratio * peri
        approx = cv2.approxPolyDP(xy_int, eps, True).reshape(-1,2).astype(np.float32)
        return (approx, mask) if return_mask else approx


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
