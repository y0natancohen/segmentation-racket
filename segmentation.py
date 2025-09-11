import argparse, time, cv2, torch, numpy as np, os
from PIL import Image
import sys
sys.path.append('rvm')
from model import MattingNetwork

def parse_args():
    p = argparse.ArgumentParser(description="RVM webcam demo")
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
    
    # Create output directory if in headless mode
    if a.headless:
        os.makedirs(a.output_dir, exist_ok=True)
        print(f"Running in headless mode. Output will be saved to: {a.output_dir}")

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

                view = np.hstack([frame, com_bgr, cv2.cvtColor((pha[0,0].cpu().numpy()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]) \
                       if a.show_alpha else com_bgr
            else:
                # Use simple OpenCV-based background removal
                com_bgr, mask = simple_background_removal(frame, bg)
                
                if a.show_alpha:
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    view = np.hstack([frame, com_bgr, mask_bgr])
                else:
                    view = com_bgr

            now = time.time(); fps = 0.9*fps + 0.1*(1.0/(now-last)); last = now
            method_text = "RVM" if use_rvm else "OpenCV"
            status_text = f"FPS:{fps:5.1f} Method:{method_text}"
            if use_rvm:
                status_text += f" dsr={a.dsr} fp16={a.fp16}"
            
            cv2.putText(view, status_text, (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(view, status_text, (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

            if a.headless:
                # Save frame in headless mode
                output_path = os.path.join(a.output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, view)
                frame_count += 1
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    print(f"Processed {frame_count} frames...")
            else:
                # Display in GUI mode
                window_title = "Real-time Segmentation Demo - q to quit"
                try:
                    cv2.imshow(window_title, view)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                except cv2.error as e:
                    print(f"GUI display error: {e}")
                    print("Switching to headless mode...")
                    a.headless = True
                    os.makedirs(a.output_dir, exist_ok=True)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
