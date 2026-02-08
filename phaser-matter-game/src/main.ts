import Phaser from "phaser";
import { GameWebSocket } from "./GameWebSocket";
import type { PolygonData } from "./types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const GAME_WIDTH = window.innerWidth;
const GAME_HEIGHT = window.innerHeight;

// Maximum number of balls alive at once (prevents runaway memory)
const MAX_BALLS = 60;

// ---------------------------------------------------------------------------
// Main Scene
// ---------------------------------------------------------------------------

export class MainScene extends Phaser.Scene {
  // Physics objects
  private balls: Phaser.Physics.Matter.Image[] = [];
  // @ts-ignore - assigned in create(), callback spawns balls as side-effect
  private ballSpawnTimer!: Phaser.Time.TimerEvent;

  // Polygon body (user silhouette)
  private polygonGraphics!: Phaser.GameObjects.Graphics;
  private polygonBody: MatterJS.BodyType | null = null;

  // Webcam
  private webcamVideo!: Phaser.GameObjects.Video;
  private videoStream: MediaStream | null = null;
  /** The raw HTMLVideoElement — needed by GameWebSocket for frame capture. */
  private rawVideoElement: HTMLVideoElement | null = null;

  // Networking
  private gameWs!: GameWebSocket;
  private latestPolygonData: PolygonData | null = null;

  // HUD
  private hudText!: Phaser.GameObjects.Text;

  constructor() {
    super("main");
  }

  // ---- Phaser lifecycle ---------------------------------------------------

  preload(): void {
    const g = this.add.graphics();
    const colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff];
    colors.forEach((color, i) => {
      g.fillStyle(color).fillCircle(12, 12, 12).generateTexture(`ball_${i}`, 24, 24);
      g.clear();
    });
    g.destroy();
  }

  create(): void {
    // ---- webcam background ------------------------------------------------
    this.setupWebcam();

    // ---- physics world ----------------------------------------------------
    this.matter.world.setBounds(0, 0, this.scale.width, this.scale.height);
    this.matter.world.setGravity(0, 1);

    // Solver iterations for crisp collisions
    this.matter.world.engine.positionIterations = 8;
    this.matter.world.engine.velocityIterations = 6;
    this.matter.world.engine.constraintIterations = 4;

    // Bouncy walls
    const walls = this.matter.world.walls;
    if (walls) {
      for (const w of [walls.top, walls.bottom, walls.left, walls.right]) {
        if (w) { w.restitution = 0.95; w.friction = 0.001; }
      }
    }

    // ---- balls ------------------------------------------------------------
    this.createBall(this.scale.width / 2, 50);
    this.ballSpawnTimer = this.time.addEvent({
      delay: 1000,
      callback: () => {
        if (this.balls.length < MAX_BALLS) {
          this.createBall(Phaser.Math.Between(50, this.scale.width - 50), 50);
        }
      },
      loop: true,
    });

    // ---- polygon graphics (drawn over video) ------------------------------
    this.polygonGraphics = this.add.graphics();
    this.polygonGraphics.setDepth(100);

    // ---- networking -------------------------------------------------------
    this.initializeWebSocket();

    // ---- HUD --------------------------------------------------------------
    this.hudText = this.add.text(10, 10, "", {
      fontSize: "14px",
      color: "#ffffff",
      backgroundColor: "rgba(0,0,0,0.6)",
      padding: { x: 8, y: 4 },
    });
    this.hudText.setScrollFactor(0);
    this.hudText.setDepth(1000);

    // ---- resize handler ---------------------------------------------------
    this.scale.on("resize", (gameSize: Phaser.Structs.Size) => {
      const { width, height } = gameSize;
      this.matter.world.setBounds(0, 0, width, height);
      this.cameras.main.setSize(width, height);
      this.fitVideoToGame(width, height);
    });
  }

  update(): void {
    // Remove off-screen balls
    this.balls = this.balls.filter((ball) => {
      if (ball.y > this.scale.height + 100) {
        ball.destroy();
        return false;
      }
      return true;
    });

    // Apply latest polygon from server
    this.applyLatestPolygon();

    // HUD
    this.updateHud();
  }

  // ---- webcam -------------------------------------------------------------

  private async setupWebcam(): Promise<void> {
    try {
      console.debug("[Main] Requesting webcam...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
      this.videoStream = stream;

      // Log track settings
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings();
      console.debug(
        "[Main] Webcam acquired: %s — %dx%d @%dfps",
        track?.label ?? "?",
        settings?.width ?? "?",
        settings?.height ?? "?",
        settings?.frameRate ?? "?",
      );

      // Create Phaser video object for display
      this.webcamVideo = this.add.video(0, 0);
      this.webcamVideo.loadMediaStream(stream);
      this.webcamVideo.play();
      this.webcamVideo.setDepth(0); // behind everything

      // Wait for video metadata then fit to game
      this.webcamVideo.on("loadeddata", () => {
        const el = this.webcamVideo.video;
        console.debug(
          "[Main] Video loadeddata: videoWidth=%d videoHeight=%d",
          el?.videoWidth ?? 0,
          el?.videoHeight ?? 0,
        );
        this.fitVideoToGame(this.scale.width, this.scale.height);
      });
      this.fitVideoToGame(this.scale.width, this.scale.height);

      // Keep a reference to the raw <video> for frame capture
      this.rawVideoElement = this.webcamVideo.video;

      // If GameWebSocket is already initialised, start frame capture
      if (this.gameWs && this.rawVideoElement) {
        console.debug("[Main] Starting frame capture (webcam ready first)");
        this.gameWs.startFrameCapture(this.rawVideoElement);
      }
    } catch (err) {
      console.error("[Main] Webcam setup failed:", err);
      this.cameras.main.setBackgroundColor(0x222222);
    }
  }

  private fitVideoToGame(w: number, h: number): void {
    if (!this.webcamVideo) return;
    const el = this.webcamVideo.video;
    if (!el) return;

    const vw = el.videoWidth || 640;
    const vh = el.videoHeight || 480;
    // Cover the game area (may crop edges)
    const scale = Math.max(w / vw, h / vh);
    this.webcamVideo.setScale(scale);
    this.webcamVideo.setPosition(w / 2, h / 2);
    this.webcamVideo.setOrigin(0.5, 0.5);
  }

  // ---- networking ---------------------------------------------------------

  private initializeWebSocket(): void {
    this.gameWs = new GameWebSocket({
      serverUrl: "ws://localhost:8765",
      jpegQuality: 0.7,
      captureRate: 15,
      captureWidth: 640,
      captureHeight: 360,
      reconnectDelay: 2000,
      maxReconnectAttempts: 0,
    });

    this.gameWs.setEventHandlers({
      onPolygonData: (data) => {
        // Single-slot: always overwrite with latest
        this.latestPolygonData = data;
      },
      onConnectionStateChange: (connected) => {
        console.log(`[Main] Segmentation server ${connected ? "connected" : "disconnected"}`);
      },
      onError: (err) => {
        console.error("[Main] GameWebSocket error:", err.message);
      },
    });

    this.gameWs.connect();

    // Start frame capture if video is already available
    if (this.rawVideoElement) {
      console.debug("[Main] Starting frame capture (WS ready first)");
      this.gameWs.startFrameCapture(this.rawVideoElement);
    }
  }

  // ---- polygon update -----------------------------------------------------

  /** Maximum vertices we allow before simplifying client-side. */
  private static readonly MAX_POLYGON_VERTS = 80;
  /** Minimum polygon area (in scaled px^2) to consider valid. */
  private static readonly MIN_POLYGON_AREA = 400;

  private applyLatestPolygon(): void {
    const data = this.latestPolygonData;
    if (!data || !data.polygon || data.polygon.length < 3) return;
    // Consume so we don't re-apply next frame
    this.latestPolygonData = null;

    const originalW = data.original_image_size[1];
    const originalH = data.original_image_size[0];
    if (originalW <= 0 || originalH <= 0) {
      console.warn("[Main] Invalid original_image_size:", data.original_image_size);
      return;
    }

    const gameW = this.scale.width;
    const gameH = this.scale.height;

    // Video display dimensions (used for coordinate mapping)
    const displayW = this.webcamVideo ? this.webcamVideo.displayWidth : gameW;
    const displayH = this.webcamVideo ? this.webcamVideo.displayHeight : gameH;
    const videoX = this.webcamVideo ? this.webcamVideo.x : gameW / 2;
    const videoY = this.webcamVideo ? this.webcamVideo.y : gameH / 2;

    const scaleX = displayW / originalW;
    const scaleY = displayH / originalH;

    // Scale vertices into video-display coordinates
    let scaled = data.polygon.map((pt) => ({
      x: pt[0] * scaleX,
      y: pt[1] * scaleY,
    }));

    console.debug(
      "[Main] applyPolygon: %d verts, original=%dx%d, display=%dx%d, scale=%.2f/%.2f",
      data.polygon.length,
      originalW,
      originalH,
      displayW,
      displayH,
      scaleX,
      scaleY,
    );

    // Simplify if too many vertices (Ramer-Douglas-Peucker on the client)
    if (scaled.length > MainScene.MAX_POLYGON_VERTS) {
      const before = scaled.length;
      scaled = MainScene.simplifyPolygon(scaled, MainScene.MAX_POLYGON_VERTS);
      console.debug(
        "[Main] Simplified polygon: %d -> %d vertices",
        before,
        scaled.length,
      );
    }

    // Validate: need at least 3 vertices
    if (scaled.length < 3) {
      console.warn("[Main] Polygon rejected: <3 vertices after simplification");
      return;
    }

    // Compute signed area to reject degenerate polygons
    const area = MainScene.polygonArea(scaled);
    if (Math.abs(area) < MainScene.MIN_POLYGON_AREA) {
      console.warn(
        "[Main] Polygon rejected: area %.1f < min %d",
        Math.abs(area),
        MainScene.MIN_POLYGON_AREA,
      );
      return;
    }

    // Ensure consistent winding (counter-clockwise for Matter.js)
    if (area > 0) scaled.reverse();

    // Bounding box & centroid
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const v of scaled) {
      if (v.x < minX) minX = v.x;
      if (v.y < minY) minY = v.y;
      if (v.x > maxX) maxX = v.x;
      if (v.y > maxY) maxY = v.y;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    // Absolute center in game coordinates
    // videoX/Y is the center of the video (origin 0.5,0.5)
    const absCx = cx - displayW / 2 + videoX;
    const absCy = cy - displayH / 2 + videoY;

    // Relative vertices for Matter.js (centred on 0,0)
    const relVerts = scaled.map((v) => ({
      x: v.x - cx,
      y: v.y - cy,
    }));

    // ---- update physics body ----
    if (this.polygonBody) {
      this.matter.world.remove(this.polygonBody);
      this.polygonBody = null;
    }

    try {
      this.polygonBody = this.matter.add.fromVertices(
        absCx, absCy, relVerts, {
          isStatic: true,
          restitution: 0.9,
          friction: 0.05,
          frictionStatic: 0.05,
        },
      );
      console.debug(
        "[Main] Physics body created at (%.0f, %.0f) with %d vertices, area=%.0f",
        absCx,
        absCy,
        relVerts.length,
        Math.abs(area),
      );
    } catch (e) {
      console.error(
        "[Main] fromVertices FAILED for %d vertices at (%.0f, %.0f):",
        relVerts.length,
        absCx,
        absCy,
        e,
      );
      // fromVertices can fail on degenerate / self-intersecting polygons
      return;
    }

    // ---- redraw visual polygon ----
    this.polygonGraphics.clear();
    this.polygonGraphics.setPosition(absCx, absCy);

    this.polygonGraphics.fillStyle(0x00ff00, 0.35);
    this.polygonGraphics.lineStyle(2, 0x00ff00, 0.8);

    this.polygonGraphics.beginPath();
    this.polygonGraphics.moveTo(relVerts[0].x, relVerts[0].y);
    for (let i = 1; i < relVerts.length; i++) {
      this.polygonGraphics.lineTo(relVerts[i].x, relVerts[i].y);
    }
    this.polygonGraphics.closePath();
    this.polygonGraphics.fillPath();
    this.polygonGraphics.strokePath();
  }

  // ---- polygon geometry helpers -------------------------------------------

  /** Signed area of a polygon (positive = clockwise). */
  private static polygonArea(verts: { x: number; y: number }[]): number {
    let area = 0;
    const n = verts.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += verts[i].x * verts[j].y;
      area -= verts[j].x * verts[i].y;
    }
    return area / 2;
  }

  /**
   * Simplify a polygon to at most `maxVerts` vertices using iterative
   * removal of the vertex that contributes the least area.
   */
  private static simplifyPolygon(
    verts: { x: number; y: number }[],
    maxVerts: number,
  ): { x: number; y: number }[] {
    const pts = verts.slice(); // copy
    while (pts.length > maxVerts) {
      // Find the vertex whose removal changes the area the least
      let minCost = Infinity;
      let minIdx = 1;
      for (let i = 0; i < pts.length; i++) {
        const prev = pts[(i - 1 + pts.length) % pts.length];
        const curr = pts[i];
        const next = pts[(i + 1) % pts.length];
        // Triangle area formed by prev-curr-next
        const cost = Math.abs(
          (prev.x * (curr.y - next.y) +
            curr.x * (next.y - prev.y) +
            next.x * (prev.y - curr.y)) / 2,
        );
        if (cost < minCost) {
          minCost = cost;
          minIdx = i;
        }
      }
      pts.splice(minIdx, 1);
    }
    return pts;
  }

  // ---- ball factory -------------------------------------------------------

  private createBall(x: number, y: number): Phaser.Physics.Matter.Image {
    const idx = Phaser.Math.Between(0, 5);
    const ball = this.matter.add.image(x, y, `ball_${idx}`);
    ball.setCircle(12);
    ball.setBounce(0.95);
    ball.setFriction(0.001);
    ball.setFrictionStatic(0.001);
    ball.setFrictionAir(0.001);
    ball.setDepth(50);
    this.balls.push(ball);
    return ball;
  }

  // ---- HUD ----------------------------------------------------------------

  private updateHud(): void {
    const ws = this.gameWs;
    const conn = ws ? (ws.isConnected ? "Connected" : "Disconnected") : "N/A";
    const polyFps = ws ? ws.polygonFps.toFixed(1) : "0";
    const sendFps = ws ? ws.frameSendFps.toFixed(1) : "0";
    const rtt = ws ? ws.roundTripMs.toFixed(0) : "-";
    const renderFps = this.game.loop.actualFps.toFixed(0);
    const ballCount = this.balls.length;

    this.hudText.setText(
      `Render: ${renderFps}fps | Seg: ${polyFps}fps | Send: ${sendFps}fps | RTT: ${rtt}ms | Balls: ${ballCount} | Server: ${conn}`,
    );
  }

  // ---- cleanup ------------------------------------------------------------

  shutdown(): void {
    if (this.gameWs) {
      this.gameWs.disconnect();
    }
    if (this.webcamVideo) {
      this.webcamVideo.stop();
      this.webcamVideo.destroy();
    }
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((t) => t.stop());
    }
  }
}

// ---------------------------------------------------------------------------
// Phaser game config
// ---------------------------------------------------------------------------
new Phaser.Game({
  type: Phaser.AUTO,
  parent: document.body,
  backgroundColor: 0x000000,
  width: GAME_WIDTH,
  height: GAME_HEIGHT,
  physics: {
    default: "matter",
    matter: {
      gravity: { x: 0, y: 1 },
    },
  },
  scene: [MainScene],
  render: {
    transparent: false,
  },
  scale: {
    mode: Phaser.Scale.RESIZE,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: GAME_WIDTH,
    height: GAME_HEIGHT,
  },
});
