#!/usr/bin/env python3
"""
GPU Profiling Script for RVM Segmentation Server

Measures true GPU utilization and compares eager vs JIT-scripted inference.
Run:
    python profile_gpu.py
    python profile_gpu.py --fp16 --n-frames 200
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from torchvision.io import decode_jpeg

from segmentation.rvm.model import MattingNetwork
from segmentation.segmentation import matte_to_polygon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_eager(model_path: str, device: torch.device, fp16: bool):
    """Load model in eager mode (current server behaviour)."""
    model = MattingNetwork("mobilenetv3").eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    if fp16 and device.type == "cuda":
        model = model.half()
    return model


def load_model_jit(model_path: str, device: torch.device, fp16: bool):
    """Load model with JIT script + freeze (official speed-test approach)."""
    model = MattingNetwork("mobilenetv3").eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    if fp16 and device.type == "cuda":
        model = model.half()
    model = torch.jit.script(model)
    model = torch.jit.freeze(model)
    return model


def to_torch_image(frame_bgr: np.ndarray, device: torch.device, fp16: bool):
    """Replicate segmentation_server.to_torch_image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device, non_blocking=True).permute(2, 0, 1).float() / 255.0
    if fp16 and device.type == "cuda":
        t = t.half()
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_model(label: str, model, device: torch.device, fp16: bool,
                    h: int, w: int, dsr: float, n_warmup: int, n_frames: int,
                    use_realistic: bool, use_gpu_decode: bool = False):
    """Run benchmark and return per-frame timing dict."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    dtype = torch.float16 if fp16 else torch.float32

    # Warmup
    rec = [None, None, None, None]
    for _ in range(n_warmup):
        dummy = torch.randn(1, 3, h, w, device=device, dtype=dtype)
        with torch.inference_mode():
            _fgr, _pha, *rec = model(dummy, *rec, dsr)
    torch.cuda.synchronize()
    rec = [None, None, None, None]  # reset recurrent state
    print(f"  Warmup: {n_warmup} frames done")

    # Prepare realistic frames (JPEG decode path) if requested
    if use_realistic:
        # Create a synthetic BGR image at target resolution, encode as JPEG
        fake_bgr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        _, jpeg_buf = cv2.imencode(".jpg", fake_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
        jpeg_bytes = jpeg_buf.tobytes()
        if use_gpu_decode:
            jpeg_tensor = torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8)

    # GPU event-based timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_frames)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_frames)]

    cpu_wall_times = []
    cpu_prep_times = []
    cpu_post_times = []

    for i in range(n_frames):
        # --- Data prep ---
        t_prep_start = time.perf_counter()
        if use_realistic and use_gpu_decode:
            # GPU decode path: nvJPEG via torchvision
            decoded = decode_jpeg(jpeg_tensor, device=device)
            if fp16:
                src = decoded.half().div_(255.0).unsqueeze(0)
            else:
                src = decoded.float().div_(255.0).unsqueeze(0)
        elif use_realistic:
            # CPU decode path: cv2.imdecode + to_torch_image
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            src = to_torch_image(frame, device, fp16)
        else:
            src = torch.randn(1, 3, h, w, device=device, dtype=dtype)
        t_prep_end = time.perf_counter()
        cpu_prep_times.append((t_prep_end - t_prep_start) * 1000)

        # --- GPU inference ---
        t_wall_start = time.perf_counter()
        start_events[i].record()
        with torch.inference_mode():
            fgr, pha, *rec = model(src, *rec, dsr)
        end_events[i].record()

        # Force sync so wall time is accurate
        torch.cuda.synchronize()
        t_wall_end = time.perf_counter()
        cpu_wall_times.append((t_wall_end - t_wall_start) * 1000)

        # --- CPU post-processing (polygon extraction) ---
        t_post_start = time.perf_counter()
        pha_np = pha[0, 0].float().cpu().numpy()
        _polygon = matte_to_polygon(pha_np, threshold=0.5, min_area=2000, epsilon_ratio=0.001)
        t_post_end = time.perf_counter()
        cpu_post_times.append((t_post_end - t_post_start) * 1000)

    # Compute GPU kernel times from events
    torch.cuda.synchronize()
    gpu_times = [start_events[i].elapsed_time(end_events[i]) for i in range(n_frames)]

    # Statistics
    def stats(arr):
        a = np.array(arr)
        return a.mean(), a.std(), np.median(a), a.min(), a.max()

    gpu_mean, gpu_std, gpu_med, gpu_min, gpu_max = stats(gpu_times)
    wall_mean, wall_std, wall_med, wall_min, wall_max = stats(cpu_wall_times)
    prep_mean, _, prep_med, _, _ = stats(cpu_prep_times)
    post_mean, _, post_med, _, _ = stats(cpu_post_times)

    total_wall_ms = sum(cpu_wall_times) + sum(cpu_prep_times) + sum(cpu_post_times)
    total_gpu_ms = sum(gpu_times)
    gpu_util_pct = (total_gpu_ms / total_wall_ms) * 100 if total_wall_ms > 0 else 0

    total_elapsed_s = total_wall_ms / 1000
    effective_fps = n_frames / total_elapsed_s if total_elapsed_s > 0 else 0

    decode_label = "GPU (nvJPEG)" if use_gpu_decode else "CPU (cv2)"
    print(f"\n  Frames: {n_frames} | Resolution: {w}x{h} | FP16: {fp16} | DSR: {dsr}")
    print(f"  Realistic pipeline: {use_realistic} | Decode: {decode_label}")
    print()
    print(f"  GPU kernel time (CUDA Events):")
    print(f"    mean={gpu_mean:.2f}ms  std={gpu_std:.2f}ms  median={gpu_med:.2f}ms")
    print(f"    min={gpu_min:.2f}ms  max={gpu_max:.2f}ms")
    print()
    print(f"  CPU wall time (inference + sync):")
    print(f"    mean={wall_mean:.2f}ms  std={wall_std:.2f}ms  median={wall_med:.2f}ms")
    print(f"    min={wall_min:.2f}ms  max={wall_max:.2f}ms")
    print()
    print(f"  CPU data prep (decode + to_tensor):")
    print(f"    mean={prep_mean:.2f}ms  median={prep_med:.2f}ms")
    print()
    print(f"  CPU post-processing (polygon extraction):")
    print(f"    mean={post_mean:.2f}ms  median={post_med:.2f}ms")
    print()
    print(f"  --- Summary ---")
    print(f"  GPU utilization (kernel / total wall): {gpu_util_pct:.1f}%")
    print(f"  CPU overhead per frame: {wall_mean - gpu_mean:.2f}ms (dispatch + sync)")
    print(f"  Effective FPS (sequential): {effective_fps:.1f}")
    print(f"  Total pipeline per frame: {prep_mean + wall_mean + post_mean:.2f}ms")

    return {
        "gpu_mean_ms": gpu_mean,
        "wall_mean_ms": wall_mean,
        "prep_mean_ms": prep_mean,
        "post_mean_ms": post_mean,
        "gpu_util_pct": gpu_util_pct,
        "effective_fps": effective_fps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile GPU utilization for RVM segmentation")
    parser.add_argument("--model_path", default="models/rvm_mobilenetv3.pth")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--dsr", type=float, default=0.25, help="Downsample ratio")
    parser.add_argument("--width", type=int, default=512, help="Input width")
    parser.add_argument("--height", type=int, default=288, help="Input height")
    parser.add_argument("--n-warmup", type=int, default=10, help="Warmup frames")
    parser.add_argument("--n-frames", type=int, default=100, help="Benchmark frames")
    parser.add_argument("--realistic", action="store_true",
                        help="Include JPEG decode + OpenCV color convert in pipeline")
    parser.add_argument("--gpu-decode", action="store_true",
                        help="Also benchmark GPU JPEG decode (nvJPEG) in realistic mode")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")

    # CUDA perf flags (same as server)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {mem / 1024**3:.1f} GB")

    # Auto-download model if needed
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    if not os.path.exists(args.model_path):
        print("Downloading RVM MobileNetV3 model weights...")
        import urllib.request
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
        urllib.request.urlretrieve(url, args.model_path)

    # --- Eager mode (current server) ---
    model_eager = load_model_eager(args.model_path, device, args.fp16)
    eager_results = benchmark_model(
        "EAGER mode (current server)",
        model_eager, device, args.fp16,
        args.height, args.width, args.dsr,
        args.n_warmup, args.n_frames, args.realistic,
    )
    del model_eager
    torch.cuda.empty_cache()

    # --- JIT scripted + frozen ---
    model_jit = load_model_jit(args.model_path, device, args.fp16)
    jit_results = benchmark_model(
        "JIT SCRIPTED + FROZEN (optimized)",
        model_jit, device, args.fp16,
        args.height, args.width, args.dsr,
        args.n_warmup, args.n_frames, args.realistic,
    )

    # --- Comparison: Eager vs JIT ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Eager vs JIT")
    print(f"{'='*60}")
    gpu_speedup = eager_results["gpu_mean_ms"] / jit_results["gpu_mean_ms"] if jit_results["gpu_mean_ms"] > 0 else 0
    wall_speedup = eager_results["wall_mean_ms"] / jit_results["wall_mean_ms"] if jit_results["wall_mean_ms"] > 0 else 0
    print(f"  GPU kernel:  {eager_results['gpu_mean_ms']:.2f}ms -> {jit_results['gpu_mean_ms']:.2f}ms ({gpu_speedup:.2f}x)")
    print(f"  Wall time:   {eager_results['wall_mean_ms']:.2f}ms -> {jit_results['wall_mean_ms']:.2f}ms ({wall_speedup:.2f}x)")
    print(f"  GPU util:    {eager_results['gpu_util_pct']:.1f}% -> {jit_results['gpu_util_pct']:.1f}%")
    print(f"  FPS:         {eager_results['effective_fps']:.1f} -> {jit_results['effective_fps']:.1f}")
    cpu_overhead_before = eager_results["wall_mean_ms"] - eager_results["gpu_mean_ms"]
    cpu_overhead_after = jit_results["wall_mean_ms"] - jit_results["gpu_mean_ms"]
    print(f"  CPU overhead:{cpu_overhead_before:.2f}ms -> {cpu_overhead_after:.2f}ms")

    # --- GPU decode comparison (when --gpu-decode and --realistic) ---
    if args.gpu_decode and args.realistic:
        jit_gpu_decode_results = benchmark_model(
            "JIT + GPU JPEG DECODE (nvJPEG)",
            model_jit, device, args.fp16,
            args.height, args.width, args.dsr,
            args.n_warmup, args.n_frames, args.realistic,
            use_gpu_decode=True,
        )

        print(f"\n{'='*60}")
        print(f"  COMPARISON: CPU decode vs GPU decode (JIT model)")
        print(f"{'='*60}")
        prep_speedup = jit_results["prep_mean_ms"] / jit_gpu_decode_results["prep_mean_ms"] if jit_gpu_decode_results["prep_mean_ms"] > 0 else 0
        total_before = jit_results["prep_mean_ms"] + jit_results["wall_mean_ms"] + jit_results["post_mean_ms"]
        total_after = jit_gpu_decode_results["prep_mean_ms"] + jit_gpu_decode_results["wall_mean_ms"] + jit_gpu_decode_results["post_mean_ms"]
        total_speedup = total_before / total_after if total_after > 0 else 0
        print(f"  Data prep:   {jit_results['prep_mean_ms']:.2f}ms -> {jit_gpu_decode_results['prep_mean_ms']:.2f}ms ({prep_speedup:.2f}x)")
        print(f"  Pipeline:    {total_before:.2f}ms -> {total_after:.2f}ms ({total_speedup:.2f}x)")
        print(f"  FPS:         {jit_results['effective_fps']:.1f} -> {jit_gpu_decode_results['effective_fps']:.1f}")
    elif args.gpu_decode:
        print("\nNote: --gpu-decode requires --realistic to have effect")

    del model_jit
    torch.cuda.empty_cache()
    print()


if __name__ == "__main__":
    main()
