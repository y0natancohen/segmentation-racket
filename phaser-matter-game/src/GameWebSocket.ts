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

import type { PolygonData, GameWebSocketConfig, GameWebSocketEvents } from "./types";

const DEFAULT_CONFIG: GameWebSocketConfig = {
  serverUrl: "ws://localhost:8765",
  jpegQuality: 0.7,
  captureRate: 15,
  captureWidth: 640,
  captureHeight: 360,
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
  private captureCanvas: OffscreenCanvas;
  private captureCtx: OffscreenCanvasRenderingContext2D;
  private captureInterval: ReturnType<typeof setInterval> | null = null;
  private isSending = false;

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

  constructor(config?: Partial<GameWebSocketConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.captureCanvas = new OffscreenCanvas(
      this.config.captureWidth,
      this.config.captureHeight,
    );
    this.captureCtx = this.captureCanvas.getContext("2d")!;
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
    const intervalMs = 1000 / this.config.captureRate;
    console.debug(
      "[GameWS] Starting frame capture at %.1f fps (interval=%dms), capture size=%dx%d",
      this.config.captureRate,
      intervalMs,
      this.config.captureWidth,
      this.config.captureHeight,
    );

    this.captureInterval = setInterval(() => {
      this._captureAndSend(video);
    }, intervalMs);
  }

  /** Stop frame capture loop. */
  stopFrameCapture(): void {
    if (this.captureInterval !== null) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
      console.debug("[GameWS] Frame capture stopped");
    }
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
      }

      console.debug(
        "[GameWS] Polygon received: %d vertices, image_size=[%s], RTT=%.0fms",
        parsed.polygon?.length ?? 0,
        parsed.original_image_size?.join("x") ?? "?",
        this._lastRoundTripMs,
      );

      this.events.onPolygonData?.(parsed);
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
      // Draw video frame to offscreen canvas (downscaled)
      this.captureCtx.drawImage(
        video,
        0, 0,
        this.config.captureWidth,
        this.config.captureHeight,
      );

      // Encode as JPEG blob
      const blob = await this.captureCanvas.convertToBlob({
        type: "image/jpeg",
        quality: this.config.jpegQuality,
      });

      // Send binary
      const buffer = await blob.arrayBuffer();
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(buffer);
        this._frameSendCount++;
        this._lastFrameSendTime = performance.now();
        console.debug(
          "[GameWS] Frame sent: %d bytes JPEG (video %dx%d -> %dx%d)",
          buffer.byteLength,
          video.videoWidth,
          video.videoHeight,
          this.config.captureWidth,
          this.config.captureHeight,
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
