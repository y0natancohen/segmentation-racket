/**
 * Tests for GameWebSocket — the unified WebSocket client that sends JPEG
 * frames and receives polygon JSON.
 */

import { GameWebSocket } from "../GameWebSocket";
import type { PolygonData } from "../types";

// ---------------------------------------------------------------------------
// Mock WebSocket
// ---------------------------------------------------------------------------

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState = MockWebSocket.OPEN;
  binaryType = "blob";
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;

  sentMessages: (string | ArrayBuffer)[] = [];

  constructor(url: string) {
    this.url = url;
    // Simulate open after microtask
    queueMicrotask(() => {
      if (this.onopen) this.onopen(new Event("open"));
    });
  }

  send(data: string | ArrayBuffer): void {
    this.sentMessages.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose(new CloseEvent("close"));
  }
}

// Install mock
(global as any).WebSocket = MockWebSocket;

// Minimal mock for OffscreenCanvas (used by GameWebSocket for frame capture)
class MockOffscreenCanvas {
  width: number;
  height: number;
  private ctx = {
    drawImage: jest.fn(),
  };

  constructor(w: number, h: number) {
    this.width = w;
    this.height = h;
  }

  getContext(_id: string) {
    return this.ctx;
  }

  convertToBlob(_opts?: any): Promise<Blob> {
    return Promise.resolve(new Blob(["fake-jpeg"], { type: "image/jpeg" }));
  }
}

(global as any).OffscreenCanvas = MockOffscreenCanvas;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GameWebSocket", () => {
  let gws: GameWebSocket;

  beforeEach(() => {
    gws = new GameWebSocket({
      serverUrl: "ws://localhost:9999",
      jpegQuality: 0.7,
      captureRate: 15,
      captureWidth: 320,
      captureHeight: 240,
      reconnectDelay: 100,
      maxReconnectAttempts: 3,
    });
  });

  afterEach(() => {
    gws.disconnect();
  });

  // ---- Connection ---------------------------------------------------------

  test("connect() opens a WebSocket to the configured URL", async () => {
    gws.connect();
    // Allow the mock's queueMicrotask to fire
    await new Promise((r) => setTimeout(r, 10));
    expect(gws.isConnected).toBe(true);
  });

  test("disconnect() closes the WebSocket", async () => {
    gws.connect();
    await new Promise((r) => setTimeout(r, 10));
    gws.disconnect();
    expect(gws.isConnected).toBe(false);
  });

  test("fires onConnectionStateChange on open and close", async () => {
    const states: boolean[] = [];
    gws.setEventHandlers({
      onConnectionStateChange: (connected) => states.push(connected),
    });
    gws.connect();
    await new Promise((r) => setTimeout(r, 10));
    gws.disconnect();
    expect(states).toEqual([true, false]);
  });

  // ---- Polygon receive ----------------------------------------------------

  test("parses incoming polygon JSON and stores as latestPolygon", async () => {
    gws.connect();
    await new Promise((r) => setTimeout(r, 10));

    const testPolygon: PolygonData = {
      polygon: [[10, 20], [30, 40], [50, 60]],
      timestamp: 1234567890.123,
      original_image_size: [480, 640],
    };

    // Grab the underlying mock WS and simulate a server message
    const ws = (gws as any).ws as MockWebSocket;
    ws.onmessage!(new MessageEvent("message", {
      data: JSON.stringify(testPolygon),
    }));

    expect(gws.latestPolygon).toEqual(testPolygon);
  });

  test("onPolygonData event fires for each received polygon", async () => {
    const received: PolygonData[] = [];
    gws.setEventHandlers({ onPolygonData: (d) => received.push(d) });

    gws.connect();
    await new Promise((r) => setTimeout(r, 10));

    const ws = (gws as any).ws as MockWebSocket;

    const p1: PolygonData = {
      polygon: [[1, 2], [3, 4], [5, 6]],
      timestamp: 1.0,
      original_image_size: [100, 200],
    };
    const p2: PolygonData = {
      polygon: [[7, 8], [9, 10], [11, 12]],
      timestamp: 2.0,
      original_image_size: [100, 200],
    };

    ws.onmessage!(new MessageEvent("message", { data: JSON.stringify(p1) }));
    ws.onmessage!(new MessageEvent("message", { data: JSON.stringify(p2) }));

    expect(received).toHaveLength(2);
    expect(received[1]).toEqual(p2);
    // latestPolygon should be the last one
    expect(gws.latestPolygon).toEqual(p2);
  });

  test("consumePolygon() returns the polygon and clears it", async () => {
    gws.connect();
    await new Promise((r) => setTimeout(r, 10));

    const ws = (gws as any).ws as MockWebSocket;
    ws.onmessage!(new MessageEvent("message", {
      data: JSON.stringify({
        polygon: [[1, 2], [3, 4], [5, 6]],
        timestamp: 1.0,
        original_image_size: [100, 200],
      }),
    }));

    const consumed = gws.consumePolygon();
    expect(consumed).not.toBeNull();
    expect(gws.latestPolygon).toBeNull();
  });

  test("invalid JSON fires onError", async () => {
    const errors: Error[] = [];
    gws.setEventHandlers({ onError: (e) => errors.push(e) });

    gws.connect();
    await new Promise((r) => setTimeout(r, 10));

    const ws = (gws as any).ws as MockWebSocket;
    ws.onmessage!(new MessageEvent("message", { data: "not-json{{{" }));

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toContain("parse");
  });
});
