/**
 * GameWebSocket — single WebSocket class that replaces both
 * VideoCommunicationManager (WebRTC) and PolygonWebSocketManager.
 *
 * Responsibilities:
 *  1. Capture frames from a <video> element as JPEG blobs.
 *  2. Send them to the Python segmentation server as binary WS messages.
 *  3. Receive polygon JSON back and expose the latest data.
 *
 * Design:
 *  - Single-slot send buffer: if a previous send is still in flight the
 *    current frame is dropped (frame-drop over accumulated latency).
 *  - Single-slot receive buffer: only the most recent polygon is kept.
 *  - Auto-reconnect with configurable delay.
 */

import type { PolygonData, GameWebSocketConfig, GameWebSocketEvents, DetailedTimingStats } from "./types";

// ---------------------------------------------------------------------------
// Rolling average helper (fixed-size ring buffer)
// ---------------------------------------------------------------------------
class RollingAverage {
  private samples: Float64Array;
  private maxSamples: number;
  private idx = 0;
  private count = 0;

  constructor(maxSamples = 60) {
    this.maxSamples = maxSamples;
    this.samples = new Float64Array(maxSamples);
  }

  push(value: number): void {
    this.samples[this.idx] = value;
    this.idx = (this.idx + 1) % this.maxSamples;
    if (this.count < this.maxSamples) this.count++;
  }

  get average(): number {
    if (this.count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.count; i++) sum += this.samples[i];
    return sum / this.count;
  }
}

const DEFAULT_CONFIG: GameWebSocketConfig = {
  serverUrl: "ws://localhost:8765",
  jpegQuality: 0.6,
  captureRate: 15,
  captureWidth: 512,
  captureHeight: 288,
  reconnectDelay: 2000,
  maxReconnectAttempts: 0, // unlimited
};

export class GameWebSocket {
  private config: GameWebSocketConfig;
  private events: GameWebSocketEvents = {};

  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private intentionallyClosed = false;

  // Frame capture
  private _captureCanvas: OffscreenCanvas;
  private captureCtx: OffscreenCanvasRenderingContext2D;
  private captureInterval: ReturnType<typeof setInterval> | null = null;

  /** Frame store: maps frame_timestamp (ms) → captured ImageData.
   *  When a polygon arrives, we look up the exact frame it was computed from. */
  private _frameStore: Map<number, ImageData> = new Map();
  /** Max frames to keep in the store before evicting oldest. */
  private static readonly MAX_FRAME_STORE = 30;
  /** requestVideoFrameCallback handle (0 = not active) */
  private rvfcHandle = 0;
  private isSending = false;
  /** Reference to the video element for rvfc re-registration */
  private _captureVideo: HTMLVideoElement | null = null;

  // Latest polygon (single-slot)
  private _latestPolygon: PolygonData | null = null;

  // Performance counters
  private _polygonCount = 0;
  private _frameSendCount = 0;
  private _frameDropCount = 0;
  private _lastCountReset = performance.now();
  private _polygonFps = 0;
  private _frameSendFps = 0;
  private _lastRoundTripMs = 0;
  /** Timestamp (performance.now) when the last frame was sent */
  private _lastFrameSendTime = 0;

  // Displayed frame/polygon timestamps (for HUD)
  private _displayedFrameTs = 0;
  private _displayedPolygonTs = 0;

  // Rolling averages for detailed timing (60-sample window)
  private _avgCapture = new RollingAverage(60);
  private _avgCaptureDraw = new RollingAverage(60);
  private _avgCaptureEncode = new RollingAverage(60);
  private _avgRtt = new RollingAverage(60);
  private _avgServerDecode = new RollingAverage(60);
  private _avgServerInference = new RollingAverage(60);
  private _avgServerPolygon = new RollingAverage(60);
  private _avgServerTotal = new RollingAverage(60);
  private _avgApply = new RollingAverage(60);

  constructor(config?: Partial<GameWebSocketConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this._captureCanvas = new OffscreenCanvas(
      this.config.captureWidth,
      this.config.captureHeight,
    );
    this.captureCtx = this._captureCanvas.getContext("2d")!;
    console.debug(
      "[GameWS] Created with config:",
      JSON.stringify(this.config),
    );
  }

  // ---- public API ---------------------------------------------------------

  /** Register event handlers. */
  setEventHandlers(events: GameWebSocketEvents): void {
    this.events = { ...this.events, ...events };
  }

  /** Connect to the segmentation server. */
  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    this.intentionallyClosed = false;
    console.debug("[GameWS] Connecting to", this.config.serverUrl);
    this._connect();
  }

  /** Disconnect and stop frame capture. */
  disconnect(): void {
    this.intentionallyClosed = true;
    this.stopFrameCapture();
    this._clearReconnect();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    console.debug("[GameWS] Disconnected (intentional)");
  }

  /** Start capturing frames from a video element and sending them. */
  startFrameCapture(video: HTMLVideoElement): void {
    this.stopFrameCapture(); // idempotent
    this._captureVideo = video;

    // Prefer requestVideoFrameCallback for frame-accurate pacing
    if ("requestVideoFrameCallback" in video) {
      console.debug(
        "[GameWS] Starting frame capture via requestVideoFrameCallback, capture size=%dx%d",
        this.config.captureWidth,
        this.config.captureHeight,
      );
      this._rvfcLoop(video);
    } else {
      // Fallback to setInterval
      const intervalMs = 1000 / this.config.captureRate;
      console.debug(
        "[GameWS] Starting frame capture at %.1f fps (interval=%dms, setInterval fallback), capture size=%dx%d",
        this.config.captureRate,
        intervalMs,
        this.config.captureWidth,
        this.config.captureHeight,
      );
      this.captureInterval = setInterval(() => {
        this._captureAndSend(video);
      }, intervalMs);
    }
  }

  /** requestVideoFrameCallback loop — re-registers itself each frame. */
  private _rvfcLoop(video: HTMLVideoElement): void {
    this.rvfcHandle = (video as any).requestVideoFrameCallback(
      (_now: number, _metadata: any) => {
        this._captureAndSend(video);
        // Re-register for next video frame
        if (this._captureVideo === video) {
          this._rvfcLoop(video);
        }
      },
    );
  }

  /** Stop frame capture loop. */
  stopFrameCapture(): void {
    if (this.captureInterval !== null) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
    if (this.rvfcHandle !== 0 && this._captureVideo) {
      (this._captureVideo as any).cancelVideoFrameCallback(this.rvfcHandle);
      this.rvfcHandle = 0;
    }
    this._captureVideo = null;
    console.debug("[GameWS] Frame capture stopped");
  }

  /** Get the latest polygon data (may be null). Does NOT consume it. */
  get latestPolygon(): PolygonData | null {
    return this._latestPolygon;
  }

  /** Consume the latest polygon (returns it and sets internal to null). */
  consumePolygon(): PolygonData | null {
    const p = this._latestPolygon;
    this._latestPolygon = null;
    return p;
  }

  /** True when the WebSocket is open. */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /** Performance: polygon messages received per second. */
  get polygonFps(): number { return this._polygonFps; }

  /** Performance: frames sent to server per second. */
  get frameSendFps(): number { return this._frameSendFps; }

  /** Performance: latest round-trip latency estimate in ms. */
  get roundTripMs(): number { return this._lastRoundTripMs; }

  /** Detailed rolling-average timing breakdown for the full pipeline. */
  get detailedTimings(): DetailedTimingStats {
    const capture_ms = this._avgCapture.average;
    const capture_draw_ms = this._avgCaptureDraw.average;
    const capture_encode_ms = this._avgCaptureEncode.average;
    const rtt_ms = this._avgRtt.average;
    const server_total_ms = this._avgServerTotal.average;
    const apply_ms = this._avgApply.average;
    const network_ms = Math.max(0, rtt_ms - server_total_ms);
    return {
      capture_ms,
      capture_draw_ms,
      capture_encode_ms,
      server_decode_ms: this._avgServerDecode.average,
      server_inference_ms: this._avgServerInference.average,
      server_polygon_ms: this._avgServerPolygon.average,
      server_total_ms,
      rtt_ms,
      network_ms,
      apply_ms,
      overall_ms: capture_ms + rtt_ms + apply_ms,
    };
  }

  /** Called by MainScene after applying a polygon to record apply time. */
  pushApplyTiming(ms: number): void {
    this._avgApply.push(ms);
  }

  /** Expose the capture canvas (current capture, may be newer than what was sent). */
  get captureCanvas(): OffscreenCanvas {
    return this._captureCanvas;
  }

  /** Frame timestamp (ms epoch) of the currently displayed frame. */
  get displayedFrameTs(): number { return this._displayedFrameTs; }

  /** Frame timestamp (ms epoch) echoed by the server for the displayed polygon. */
  get displayedPolygonTs(): number { return this._displayedPolygonTs; }

  // ---- internals ----------------------------------------------------------

  private _connect(): void {
    try {
      this.ws = new WebSocket(this.config.serverUrl);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        console.debug("[GameWS] WebSocket OPEN to", this.config.serverUrl);
        this.events.onConnectionStateChange?.(true);
      };

      this.ws.onmessage = (ev: MessageEvent) => {
        if (typeof ev.data === "string") {
          console.debug(
            "[GameWS] Received text message (%d chars)",
            ev.data.length,
          );
          this._handleTextMessage(ev.data);
        } else {
          console.debug(
            "[GameWS] Received unexpected binary message (%d bytes), ignoring",
            ev.data instanceof ArrayBuffer ? ev.data.byteLength : 0,
          );
        }
      };

      this.ws.onclose = (ev) => {
        console.debug(
          "[GameWS] WebSocket CLOSED code=%d reason=%s wasClean=%s",
          ev.code,
          ev.reason || "(none)",
          ev.wasClean,
        );
        this.events.onConnectionStateChange?.(false);
        if (!this.intentionallyClosed) this._scheduleReconnect();
      };

      this.ws.onerror = (ev) => {
        console.warn("[GameWS] WebSocket ERROR", ev);
        this.events.onError?.(new Error("WebSocket error"));
      };
    } catch (err) {
      console.error("[GameWS] Failed to create WebSocket:", err);
      this.events.onError?.(
        err instanceof Error ? err : new Error(String(err)),
      );
      if (!this.intentionallyClosed) this._scheduleReconnect();
    }
  }

  private _handleTextMessage(data: string): void {
    try {
      const parsed = JSON.parse(data) as PolygonData;
      this._latestPolygon = parsed; // single-slot overwrite
      this._polygonCount++;

      // Round-trip latency estimate
      if (this._lastFrameSendTime > 0) {
        this._lastRoundTripMs = performance.now() - this._lastFrameSendTime;
        this._avgRtt.push(this._lastRoundTripMs);
      }

      // Accumulate server-side timings
      const st = parsed.server_timings;
      if (st) {
        this._avgServerDecode.push(st.decode_ms);
        this._avgServerInference.push(st.inference_ms);
        this._avgServerPolygon.push(st.polygon_ms);
        this._avgServerTotal.push(st.total_ms);
      }

      // ---- Frame-polygon correlation via timestamp ----
      const frameTs = parsed.frame_timestamp ?? 0;
      let frameImageData: ImageData | null = null;
      if (frameTs && this._frameStore.has(frameTs)) {
        frameImageData = this._frameStore.get(frameTs)!;
        this._frameStore.delete(frameTs);
      }
      // Evict stale entries older than 2 seconds
      const cutoff = Date.now() - 2000;
      for (const key of this._frameStore.keys()) {
        if (key < cutoff) this._frameStore.delete(key);
      }
      // Track displayed timestamps
      this._displayedFrameTs = frameTs;
      this._displayedPolygonTs = frameTs;

      console.debug(
        "[GameWS] Polygon received: %d vertices, frame_ts=%d, matched=%s, RTT=%.0fms",
        parsed.polygon?.length ?? 0,
        frameTs,
        frameImageData !== null,
        this._lastRoundTripMs,
      );

      this.events.onPolygonData?.(parsed, frameImageData);
    } catch (e) {
      console.warn("[GameWS] Failed to parse polygon JSON:", e, "raw:", data.slice(0, 200));
      this.events.onError?.(new Error("Failed to parse polygon JSON"));
    }
  }

  private async _captureAndSend(video: HTMLVideoElement): Promise<void> {
    // Drop frame if previous send still in flight or not connected
    if (this.isSending) {
      this._frameDropCount++;
      console.debug("[GameWS] Frame DROPPED (still sending previous). Total drops=%d", this._frameDropCount);
      return;
    }
    if (!this.isConnected) {
      console.debug("[GameWS] Frame DROPPED (not connected)");
      return;
    }

    // Ensure video has data
    if (video.readyState < 2) { // HAVE_CURRENT_DATA
      console.debug(
        "[GameWS] Frame DROPPED (video not ready, readyState=%d)",
        video.readyState,
      );
      return;
    }

    this.isSending = true;
    try {
      const captureStart = performance.now();
      const frameTs = Date.now(); // Integer ms — unique ID for this frame

      // Draw video frame to offscreen canvas (downscaled)
      this.captureCtx.drawImage(
        video,
        0, 0,
        this.config.captureWidth,
        this.config.captureHeight,
      );

      const drawEnd = performance.now();
      const drawMs = drawEnd - captureStart;
      this._avgCaptureDraw.push(drawMs);

      // Snapshot frame pixels for later correlation with the polygon
      const imageData = this.captureCtx.getImageData(
        0, 0, this.config.captureWidth, this.config.captureHeight,
      );
      this._frameStore.set(frameTs, imageData);
      // Evict oldest if store is too large
      if (this._frameStore.size > GameWebSocket.MAX_FRAME_STORE) {
        const oldest = this._frameStore.keys().next().value;
        if (oldest !== undefined) this._frameStore.delete(oldest);
      }

      // Encode as JPEG blob
      const blob = await this._captureCanvas.convertToBlob({
        type: "image/jpeg",
        quality: this.config.jpegQuality,
      });

      const jpegBuffer = await blob.arrayBuffer();

      const encodeMs = performance.now() - drawEnd;
      this._avgCaptureEncode.push(encodeMs);

      const captureMs = performance.now() - captureStart;
      this._avgCapture.push(captureMs);

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        // Prepend 8-byte Float64 timestamp header to JPEG data
        const header = new ArrayBuffer(8);
        new DataView(header).setFloat64(0, frameTs);
        const combined = new Uint8Array(8 + jpegBuffer.byteLength);
        combined.set(new Uint8Array(header), 0);
        combined.set(new Uint8Array(jpegBuffer), 8);

        this.ws.send(combined.buffer);
        this._frameSendCount++;
        this._lastFrameSendTime = performance.now();
        console.debug(
          "[GameWS] Frame sent: ts=%d, %d bytes JPEG (video %dx%d -> %dx%d) capture=%.1fms",
          frameTs,
          jpegBuffer.byteLength,
          video.videoWidth,
          video.videoHeight,
          this.config.captureWidth,
          this.config.captureHeight,
          captureMs,
        );
        this.events.onFrameSent?.();
      }
    } catch (e) {
      console.warn("[GameWS] Frame capture/send error:", e);
      // Silently drop on error (frame drop policy)
    } finally {
      this.isSending = false;
    }

    // Update FPS counters every second
    const now = performance.now();
    if (now - this._lastCountReset >= 1000) {
      const elapsed = (now - this._lastCountReset) / 1000;
      this._polygonFps = this._polygonCount / elapsed;
      this._frameSendFps = this._frameSendCount / elapsed;
      this._polygonCount = 0;
      this._frameSendCount = 0;
      this._frameDropCount = 0;
      this._lastCountReset = now;
    }
  }

  private _scheduleReconnect(): void {
    this._clearReconnect();
    const max = this.config.maxReconnectAttempts;
    if (max > 0 && this.reconnectAttempts >= max) return;

    this.reconnectAttempts++;
    console.debug(
      "[GameWS] Scheduling reconnect in %dms (attempt %d)",
      this.config.reconnectDelay,
      this.reconnectAttempts,
    );
    this.reconnectTimer = setTimeout(() => {
      this._connect();
    }, this.config.reconnectDelay);
  }

  private _clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
