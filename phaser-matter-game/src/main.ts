import Phaser from "phaser";
import { VideoCommunicationManager } from "./VideoCommunicationManager";
import { PolygonWebSocketManager } from "./PolygonWebSocketManager";
import type { VideoConfig, PolygonData } from "./types";

// Game dimensions constants - use full window size
const GAME_WIDTH = window.innerWidth;
const GAME_HEIGHT = window.innerHeight;

interface PolygonMessage {
  position: { x: number; y: number };
  vertices: { x: number; y: number }[];
  rotation: number;
}

export class MainScene extends Phaser.Scene {
  private balls: Phaser.Physics.Matter.Image[] = [];
  // @ts-ignore - Used for timer callback side effects
  private ballSpawnTimer!: Phaser.Time.TimerEvent;
  private platform!: Phaser.GameObjects.Graphics;
  private platformBody!: MatterJS.BodyType;
  
  // Video communication and polygon WebSocket managers
  private videoManager!: VideoCommunicationManager;
  private polygonManager!: PolygonWebSocketManager;
  private connectionId: string = 'phaser_game_connection';
  
  // Latest message storage
  private latestPolygonData: PolygonData | null = null;
  
  // FPS monitoring
  private fpsText!: Phaser.GameObjects.Text;
  private statusText!: Phaser.GameObjects.Text;
  private videoStatusText!: Phaser.GameObjects.Text;
  private messageCount: number = 0;
  private lastFpsUpdateTime: number = 0;
  private currentFps: number = 0;
  
  // Webcam integration
  private webcamVideo!: Phaser.GameObjects.Video;
  private videoStream: MediaStream | null = null;

  constructor() { super("main"); }

  preload() {
    // Generate colored ball textures and black platform
    const g = this.add.graphics();
    
    // Create different colored balls
    const colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff];
    colors.forEach((color, index) => {
      g.fillStyle(color).fillCircle(12, 12, 12).generateTexture(`ball_${index}`, 24, 24);
      g.clear();
    });
    
    // Create platform texture
    g.fillStyle(0x000000).fillRect(0, 0, 200, 12).generateTexture("plat", 200, 12);
    g.destroy();
  }

  create() {
    console.log("=== CREATE METHOD CALLED ===");
    console.log("Game scale dimensions:", this.scale.width, this.scale.height);
    
    // Test: Create a simple rectangle to see if basic rendering works
    const testRect = this.add.rectangle(100, 100, 100, 100, 0x00ff00);
    testRect.setDepth(1000);
    
    // Add visual debugging text instead of console logging
    this.add.text(10, 10, "Game Started - Console logging not working", {
      fontSize: '16px',
      color: '#ffffff',
      backgroundColor: '#000000',
      padding: { x: 8, y: 4 }
    }).setDepth(2000);
    
    // Set up webcam background using Phaser Video Game Object
    this.setupWebcamBackground();
    
    // Handle window resize using Phaser scale system
    this.scale.on('resize', (gameSize: any) => {
      const { width, height } = gameSize;
      console.log('Window resized to:', width, 'x', height);
      
      // Update physics world bounds
      this.matter.world.setBounds(0, 0, width, height);
      
      // Update camera size properly
      this.cameras.main.setSize(width, height);
      
      // Resize video using the single method
      this.resizeVideoToGameSize(width, height);
      
      // Ensure all game objects are still visible and properly positioned
      this.cameras.main.setBackgroundColor(0x000000);
      
      // Ensure text objects are still visible
      if (this.fpsText) {
        this.fpsText.setScrollFactor(0);
        this.fpsText.setDepth(1000);
      }
      if (this.statusText) {
        this.statusText.setScrollFactor(0);
        this.statusText.setDepth(1000);
      }
      
      // Ensure platform is still visible
      if (this.platform) {
        this.platform.setDepth(100);
      }
      
      // Ensure balls are still visible
      this.balls.forEach(ball => {
        if (ball && ball.body) {
          ball.setDepth(50);
        }
      });
    });

    // Enable Matter physics
    this.matter.world.setBounds(0, 0, this.scale.width, this.scale.height);
    this.matter.world.setGravity(0, 1); // gravity down

    // Raise solver iterations for more accurate, less "mushy" collisions
    this.matter.world.engine.positionIterations = 8; // default ~6
    this.matter.world.engine.velocityIterations = 6; // default ~4
    this.matter.world.engine.constraintIterations = 4; // default ~2

    // Create initial ball
    this.createBall(this.scale.width / 2, 50);

    // Set up timer to spawn new balls every second
    this.ballSpawnTimer = this.time.addEvent({
      delay: 1000, // 1 second
      callback: () => {
        // Spawn ball at random x position near the top
        const x = Phaser.Math.Between(50, this.scale.width - 50);
        this.createBall(x, 50);
      },
      loop: true
    });

    // Floor (bottom of the square) with bouncy physics
    this.matter.add.rectangle(this.scale.width / 2, this.scale.height - 10, this.scale.width - 20, 20, { 
      isStatic: true,
      restitution: 0.95,
      friction: 0.001,
      frictionStatic: 0.001
    });
    
    // Make walls bouncy for better elastic collisions
    const walls = this.matter.world.walls; // created by setBounds
    if (walls) {
      // Make every wall bouncy as well
      if (walls.top) walls.top.restitution = 0.95;
      if (walls.bottom) walls.bottom.restitution = 0.95;
      if (walls.left) walls.left.restitution = 0.95;
      if (walls.right) walls.right.restitution = 0.95;

      // Minimal wall friction for realistic physics
      if (walls.top) walls.top.friction = 0.001;
      if (walls.bottom) walls.bottom.friction = 0.001;
      if (walls.left) walls.left.friction = 0.001;
      if (walls.right) walls.right.friction = 0.001;
    }

    // Create initial polygon platform (will be updated from WebSocket data)
    this.createPolygonPlatform(this.scale.width / 2, this.scale.height * 0.7);

    // Initialize video communication and polygon WebSocket
    this.initializeVideoCommunication();
    this.initializePolygonWebSocket();

    // Initialize FPS tracking
    this.messageCount = 0;
    this.lastFpsUpdateTime = this.time.now;
    this.currentFps = 0;

    // Add FPS display
    this.fpsText = this.add.text(10, 10, "Rectangle FPS: 0.0", {
      fontSize: '16px',
      color: '#000000',
      backgroundColor: '#ffffff',
      padding: { x: 8, y: 4 }
    });
    this.fpsText.setScrollFactor(0); // Keep text fixed on screen
    this.fpsText.setDepth(1000); // Ensure it's on top

    // Add status display
    this.statusText = this.add.text(10, 35, "Polygon: Disconnected", {
      fontSize: '14px',
      color: '#ffffff',
      backgroundColor: '#ff0000',
      padding: { x: 8, y: 4 }
    });
    this.statusText.setScrollFactor(0); // Keep text fixed on screen
    this.statusText.setDepth(1000); // Ensure it's on top

    // Add video status display
    this.videoStatusText = this.add.text(10, 60, "Video: Disconnected", {
      fontSize: '14px',
      color: '#ffffff',
      backgroundColor: '#ff0000',
      padding: { x: 8, y: 4 }
    });
    this.videoStatusText.setScrollFactor(0); // Keep text fixed on screen
    this.videoStatusText.setDepth(1000); // Ensure it's on top

  }

  update() {
    // Clean up balls that have fallen off the screen
    this.balls = this.balls.filter(ball => {
      if (ball.y > this.scale.height + 100) { // If ball is below the screen
        ball.destroy();
        return false; // Remove from array
      }
      return true; // Keep in array
    });

    this.updatePlatformFromPolygonData();
    
    this.updateFpsDisplay();
    
    this.updateStatusDisplay();
  }

  private createPolygonPlatform(x: number, y: number): void {
    // Create a graphics object to draw the polygon
    this.platform = this.add.graphics();
    this.platform.setPosition(x, y);
    this.platform.setDepth(100); // Ensure polygon is above video (depth 0)
    
    // Create initial Matter.js body for physics (will be updated with real vertices)
    const initialVertices = [
      { x: -50, y: -25 },
      { x: 50, y: -25 },
      { x: 50, y: 25 },
      { x: -50, y: 25 }
    ];
    
    // Create physics body from vertices
    this.platformBody = this.matter.add.fromVertices(x, y, initialVertices, { 
      isStatic: true,
      restitution: 0.95,
      friction: 0.001,
      frictionStatic: 0.001
    });
    
    // Draw initial rectangle
    this.drawPolygon(initialVertices);
  }

  private updatePolygonPlatform(message: PolygonMessage): void {
    if (!this.platform) return;

    console.log('=== UPDATING POLYGON PLATFORM ===');
    console.log('Platform position:', message.position);
    console.log('Platform vertices:', message.vertices);
    console.log('Platform rotation:', message.rotation);

    // Update graphics position and rotation
    this.platform.setPosition(message.position.x, message.position.y);
    this.platform.setRotation(message.rotation);
    
    // Remove old physics body
    if (this.platformBody) {
      this.matter.world.remove(this.platformBody);
    }
    
    // Create new physics body with updated vertices
    this.platformBody = this.matter.add.fromVertices(
      message.position.x, 
      message.position.y, 
      message.vertices, 
      { 
        isStatic: true,
        restitution: 0.95,
        friction: 0.001,
        frictionStatic: 0.001
      }
    );
    
    // Set rotation on the physics body
    this.matter.body.setAngle(this.platformBody, message.rotation);
    
    // Clear and redraw the polygon with new vertices
    this.platform.clear();
    this.drawPolygon(message.vertices);
    
    console.log('Platform updated successfully');
    console.log('=== END PLATFORM UPDATE ===');
  }
  
  private drawPolygon(vertices: { x: number; y: number }[]): void {
    if (!this.platform || vertices.length < 3) return;
    
    // Set fill color and line style
    this.platform.fillStyle(0x00ff00, 0.8); // Green with transparency
    this.platform.lineStyle(2, 0x00aa00, 1); // Darker green border
    
    // Start drawing the polygon
    this.platform.beginPath();
    this.platform.moveTo(vertices[0].x, vertices[0].y);
    
    // Draw lines to each vertex
    for (let i = 1; i < vertices.length; i++) {
      this.platform.lineTo(vertices[i].x, vertices[i].y);
    }
    
    // Close the polygon
    this.platform.closePath();
    this.platform.fillPath();
    this.platform.strokePath();
  }

  private createBall(x: number, y: number): Phaser.Physics.Matter.Image {
    // Choose random colored ball
    const ballColorIndex = Phaser.Math.Between(0, 5);
    const ballTexture = `ball_${ballColorIndex}`;
    
    const ball = this.matter.add.image(x, y, ballTexture);
    ball.setCircle(12);                // circular collider
    ball.setBounce(0.95);              // restitution ~1 ⇒ very bouncy
    ball.setFriction(0.001);           // surface friction
    ball.setFrictionStatic(0.001);     // static friction
    ball.setFrictionAir(0.001);        // tiny air drag, prevents "infinite" jitter
    
    // Set angular damping through the body
    if (ball.body && 'angularDamping' in ball.body) {
      (ball.body as any).angularDamping = 0.001; // tiny spin damp
    }
    
    // Add to balls array for tracking
    this.balls.push(ball);
    
    return ball;
  }

  private async initializeVideoCommunication(): Promise<void> {
    try {
      console.log("Initializing video communication...");
      
      // Create video configuration
      const videoConfig: VideoConfig = {
        serverUrl: 'http://localhost:8080',
        videoConstraints: {
          width: { ideal: GAME_WIDTH, max: 1920 },
          height: { ideal: GAME_HEIGHT, max: 1080 },
          frameRate: { ideal: 30, max: 60 },
          facingMode: 'user'
        },
        dataChannelConfig: {
          ordered: true,
          maxRetransmits: 3,
          protocol: 'metrics'
        },
        reconnectInterval: 2000,
        maxReconnectAttempts: 5
      };

      // Initialize video manager
      this.videoManager = new VideoCommunicationManager(videoConfig);
      
      // Set up event handlers
      this.videoManager.setEventHandlers({
        onConnectionStateChange: (_connectionId, state) => {
          console.log(`Video connection state changed: ${state}`);
          this.updateVideoStatusDisplay();
        },
        onStreamReady: async (_connectionId, stream) => {
          console.log("Video stream ready, setting up webcam background");
          this.videoStream = stream;
          await this.setupWebcamBackground(stream);
        },
        onIntensityUpdate: (_connectionId, metrics) => {
          console.log(`Intensity update: ${metrics.avg_intensity.toFixed(1)}`);
        },
        onError: (_connectionId, error) => {
          console.error("Video communication error:", error);
        }
      });

      // Create connection and start camera
      await this.videoManager.createConnection(this.connectionId);
      const stream = await this.videoManager.startCamera(this.connectionId);
      await this.videoManager.connect(this.connectionId, stream);
      
      console.log("Video communication initialized successfully");
    } catch (error) {
      console.error("Failed to initialize video communication:", error);
    }
  }

  private async initializePolygonWebSocket(): Promise<void> {
    try {
      console.log("Initializing polygon WebSocket...");
      
      // Initialize polygon manager
      this.polygonManager = new PolygonWebSocketManager('ws://localhost:8080/polygon');
      
      // Set up event handlers
      this.polygonManager.setEventHandlers({
        onPolygonData: (data) => {
          console.log("Polygon data received:", data);
          this.latestPolygonData = data;
          this.messageCount++;
        },
        onConnectionStateChange: (connected) => {
          console.log(`Polygon WebSocket ${connected ? 'connected' : 'disconnected'}`);
        this.updateStatusDisplay();
        },
        onError: (error) => {
          console.error("Polygon WebSocket error:", error);
        }
      });

      // Connect to polygon WebSocket
      await this.polygonManager.connect();
      
      console.log("Polygon WebSocket initialized successfully");
    } catch (error) {
      console.error("Failed to initialize polygon WebSocket:", error);
    }
  }

  private updatePlatformFromPolygonData(): void {
    if (this.latestPolygonData) {
      // Convert polygon data to PolygonMessage format
      const polygonMessage = this.convertPolygonDataToMessage(this.latestPolygonData);
      this.updatePolygonPlatform(polygonMessage);
    }
  }

  private convertPolygonDataToMessage(polygonData: PolygonData): PolygonMessage {
    // Get the original image dimensions from the polygon data
    const originalWidth = polygonData.original_image_size[1]; // width
    const originalHeight = polygonData.original_image_size[0]; // height
    
    // Get current game/video dimensions
    const gameWidth = this.scale.width;
    const gameHeight = this.scale.height;
    
    // Get video position and scale information
    const videoX = this.webcamVideo ? this.webcamVideo.x : 0;
    const videoY = this.webcamVideo ? this.webcamVideo.y : 0;
    const videoScaleX = this.webcamVideo ? this.webcamVideo.scaleX : 1;
    const videoScaleY = this.webcamVideo ? this.webcamVideo.scaleY : 1;
    const videoDisplayWidth = this.webcamVideo ? this.webcamVideo.displayWidth : gameWidth;
    const videoDisplayHeight = this.webcamVideo ? this.webcamVideo.displayHeight : gameHeight;
    
    console.log('=== POLYGON COORDINATE TRANSFORMATION ===');
    console.log(`Original image size: ${originalWidth}x${originalHeight}`);
    console.log(`Game dimensions: ${gameWidth}x${gameHeight}`);
    console.log(`Video position: (${videoX}, ${videoY})`);
    console.log(`Video scale: ${videoScaleX}x${videoScaleY}`);
    console.log(`Video display size: ${videoDisplayWidth}x${videoDisplayHeight}`);
    console.log(`Raw polygon points:`, polygonData.polygon);
    
    // Calculate scaling factors based on video display size vs original image size
    const scaleX = videoDisplayWidth / originalWidth;
    const scaleY = videoDisplayHeight / originalHeight;
    
    console.log(`Calculated scale factors: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`);
    
    // Scale the polygon vertices to match the video display dimensions (NO video offset yet)
    const scaledVertices = polygonData.polygon.map((point, index) => {
      const scaledX = point[0] * scaleX;
      const scaledY = point[1] * scaleY;
      
      console.log(`Point ${index}: (${point[0]}, ${point[1]}) -> scaled: (${scaledX.toFixed(1)}, ${scaledY.toFixed(1)})`);
      
      return {
        x: scaledX,
        y: scaledY
      };
    });
    
    // Calculate the bounding box of the polygon to find top-left corner
    const minX = Math.min(...scaledVertices.map(v => v.x));
    const minY = Math.min(...scaledVertices.map(v => v.y));
    const maxX = Math.max(...scaledVertices.map(v => v.x));
    const maxY = Math.max(...scaledVertices.map(v => v.y));
    
    // Calculate polygon dimensions
    const polygonWidth = maxX - minX;
    const polygonHeight = maxY - minY;
    
    // Calculate the center of the detected object in the video
    const detectedCenterX = minX + (polygonWidth / 2);
    const detectedCenterY = minY + (polygonHeight / 2);
    
    // Position the polygon center at the detected object center in video coordinates
    const centerX = detectedCenterX + videoX;
    const centerY = detectedCenterY + videoY;
    
    // Convert vertices to be RELATIVE to the polygon center (for proper centering)
    const relativeVertices = scaledVertices.map(vertex => ({
      x: vertex.x - detectedCenterX,
      y: vertex.y - detectedCenterY
    }));
    
    console.log(`Polygon bounding box: (${minX.toFixed(1)}, ${minY.toFixed(1)}) to (${maxX.toFixed(1)}, ${maxY.toFixed(1)})`);
    console.log(`Polygon dimensions: ${polygonWidth.toFixed(1)} x ${polygonHeight.toFixed(1)}`);
    console.log(`Detected object center: (${detectedCenterX.toFixed(1)}, ${detectedCenterY.toFixed(1)})`);
    console.log(`Video position: (${videoX.toFixed(1)}, ${videoY.toFixed(1)})`);
    console.log(`Final polygon position: (${centerX.toFixed(1)}, ${centerY.toFixed(1)})`);
    console.log(`Relative vertices (first 3):`, relativeVertices.slice(0, 3));
    
    console.log(`Final polygon center: (${centerX.toFixed(1)}, ${centerY.toFixed(1)})`);
    console.log(`Final vertices (relative):`, relativeVertices);
    console.log('=== END POLYGON TRANSFORMATION ===');
    
    return {
      position: { x: centerX, y: centerY },
      vertices: relativeVertices, // Use relative vertices!
      rotation: 0 // Could be calculated from polygon orientation if needed
    };
  }

  private updateFpsDisplay(): void {
    const currentTime = this.time.now;
    
    // Update FPS every second
    if (currentTime - this.lastFpsUpdateTime >= 1000) {
      this.currentFps = this.messageCount;
      this.messageCount = 0; // Reset counter
      this.lastFpsUpdateTime = currentTime;
      
      // Update FPS display
      this.fpsText.setText(`Rectangle FPS: ${this.currentFps.toFixed(1)}`);
      
      // Update status display
      this.updateStatusDisplay();
    }
  }

  private updateStatusDisplay(): void {
    if (this.polygonManager && this.polygonManager.isConnected()) {
      const hasData = this.latestPolygonData ? 'Yes' : 'No';
      this.statusText.setText(`Polygon: Connected | Data: ${hasData}`);
      this.statusText.setStyle({ color: '#000000', backgroundColor: '#00ff00' });
    } else {
      this.statusText.setText("Polygon: Disconnected");
      this.statusText.setStyle({ color: '#ffffff', backgroundColor: '#ff0000' });
    }
  }

  private updateVideoStatusDisplay(): void {
    if (this.videoManager && this.videoManager.isConnectionActive(this.connectionId)) {
      this.videoStatusText.setText("Video: Connected");
      this.videoStatusText.setStyle({ color: '#000000', backgroundColor: '#00ff00' });
    } else {
      this.videoStatusText.setText("Video: Disconnected");
      this.videoStatusText.setStyle({ color: '#ffffff', backgroundColor: '#ff0000' });
    }
  }

  private async setupWebcamBackground(stream?: MediaStream): Promise<void> {
    try {
      console.log("Setting up webcam background...");
      
      // Use provided stream or request new one
      let videoStream = stream;
      if (!videoStream) {
        console.log("Requesting webcam access...");
        videoStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: GAME_WIDTH },
          height: { ideal: GAME_HEIGHT },
          facingMode: 'user' // Front-facing camera
        } 
      });
      }
      
      console.log("Webcam stream obtained:", videoStream);
      console.log("About to create video object...");
      
      // Create Phaser Video Game Object
      this.webcamVideo = this.add.video(0, 0); // Will be positioned by configureVideoLayout
      
      // Load the media stream into the Video object
      this.webcamVideo.loadMediaStream(videoStream);
      
      // Play the video
      this.webcamVideo.play();
      
      // Wait for video to be ready before configuring layout
      this.webcamVideo.on('loadeddata', () => {
        console.log('Video data loaded, configuring layout...');
        this.configureVideoLayout();
      });
      
      // Also configure immediately in case the event doesn't fire
      this.configureVideoLayout();
      
      console.log("Webcam background set up successfully using Phaser Video Game Object");
      
    } catch (error) {
      console.error("Failed to initialize webcam:", error);
      
      // Fallback to white background if webcam fails
      this.cameras.main.setBackgroundColor(0xffffff);
    }
  }

  private configureVideoLayout(): void {
    if (!this.webcamVideo) return;
    
    // Get current game dimensions
    const gameWidth = this.scale.width;
    const gameHeight = this.scale.height;
    
    console.log('Configuring video layout for dimensions:', gameWidth, 'x', gameHeight);
    
    // Store current video state
    const wasPlaying = this.webcamVideo.isPlaying();
    
    // Set video origin to top-left (0, 0) for proper full-screen positioning
    this.webcamVideo.setOrigin(0, 0);
    
    // Position video at top-left corner (0, 0) since origin is now top-left
    this.webcamVideo.setPosition(0, 0);
    
    // Set depth to be behind all other objects
    this.webcamVideo.setDepth(0);
    
    // Ensure video is visible and active
    this.webcamVideo.setVisible(true);
    this.webcamVideo.setActive(true);
    
    // Restore video playback state if it was playing
    if (wasPlaying && !this.webcamVideo.isPlaying()) {
      this.webcamVideo.play();
    }
    
    // Use the single video sizing method
    this.resizeVideoToGameSize(gameWidth, gameHeight);
    
    // Use multiple logging methods to ensure visibility
    console.log('=== VIDEO CONFIGURATION DEBUG ===');
    console.log('Video configured - Position:', this.webcamVideo.x, this.webcamVideo.y);
    console.log('Video configured - Size:', this.webcamVideo.displayWidth, this.webcamVideo.displayHeight);
    console.log('Video configured - Depth:', this.webcamVideo.depth);
    console.log('Game dimensions:', gameWidth, 'x', gameHeight);
    console.log('Video is playing:', this.webcamVideo.isPlaying());
    console.log('Video scale:', this.webcamVideo.scaleX, this.webcamVideo.scaleY);
    console.log('Video width/height:', this.webcamVideo.width, this.webcamVideo.height);
    console.log('Video origin:', this.webcamVideo.originX, this.webcamVideo.originY);
    console.log('Video bounds:', this.webcamVideo.getBounds());
    console.log('=== END VIDEO CONFIGURATION ===');
    
    // Try DOM-based logging as well
    const debugDiv = document.createElement('div');
    debugDiv.style.position = 'fixed';
    debugDiv.style.top = '10px';
    debugDiv.style.left = '10px';
    debugDiv.style.backgroundColor = 'rgba(0,0,0,0.8)';
    debugDiv.style.color = 'white';
    debugDiv.style.padding = '10px';
    debugDiv.style.zIndex = '9999';
    debugDiv.innerHTML = `Video: ${this.webcamVideo.x}, ${this.webcamVideo.y} | Size: ${this.webcamVideo.displayWidth}x${this.webcamVideo.displayHeight} | Game: ${gameWidth}x${gameHeight} | Playing: ${this.webcamVideo.isPlaying()}`;
    document.body.appendChild(debugDiv);
    
    // Remove debug div after 3 seconds
    setTimeout(() => {
      if (debugDiv.parentNode) {
        debugDiv.parentNode.removeChild(debugDiv);
      }
    }, 3000);
  }

  private resizeVideoToGameSize(width: number, height: number): void {
    if (!this.webcamVideo) return;
    
    console.log('Resizing video to game size:', width, 'x', height);
    
    // Based on research: Phaser Video objects loaded from loadMediaStream() have limitations
    // with setDisplaySize(). We need to use setScale() instead.
    
    // Get the original video dimensions
    const videoElement = this.webcamVideo.video;
    if (videoElement) {
      const originalWidth = videoElement.videoWidth || videoElement.width || 640;
      const originalHeight = videoElement.videoHeight || videoElement.height || 480;
      
      console.log('Original video dimensions:', originalWidth, 'x', originalHeight);
      
      // Calculate scale factors to fill the game area
      const scaleX = width / originalWidth;
      const scaleY = height / originalHeight;
      
      // Use the larger scale to ensure the video covers the entire area
      const scale = Math.max(scaleX, scaleY);
      
      console.log('Calculated scale factors - X:', scaleX, 'Y:', scaleY, 'Final scale:', scale);
      
      // Apply the scale to the video
      this.webcamVideo.setScale(scale);
      
      // Center the video if it's larger than the game area
      this.webcamVideo.setPosition(width / 2, height / 2);
      this.webcamVideo.setOrigin(0.5, 0.5);
      
      // Also try direct DOM manipulation as backup
      videoElement.style.width = width + 'px';
      videoElement.style.height = height + 'px';
      videoElement.style.objectFit = 'cover';
      videoElement.style.position = 'absolute';
      videoElement.style.top = '0px';
      videoElement.style.left = '0px';
      videoElement.style.zIndex = '-1';
    } else {
      // Fallback: try Phaser methods
      this.webcamVideo.setDisplaySize(width, height);
      this.webcamVideo.setPosition(0, 0);
      this.webcamVideo.setOrigin(0, 0);
    }
    
    console.log('Video resized - Final scale:', this.webcamVideo.scaleX, this.webcamVideo.scaleY);
    console.log('Video resized - Position:', this.webcamVideo.x, this.webcamVideo.y);
    console.log('Video resized - Display size:', this.webcamVideo.displayWidth, 'x', this.webcamVideo.displayHeight);
  }

  shutdown(): void {
    // Clean up video communication
    if (this.videoManager) {
      this.videoManager.cleanup();
    }
    
    // Clean up polygon WebSocket
    if (this.polygonManager) {
      this.polygonManager.disconnect();
    }
    
    // Clean up webcam
    if (this.webcamVideo) {
      this.webcamVideo.stop();
      this.webcamVideo.destroy();
    }
    
    // Clean up video stream
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }
}

new Phaser.Game({
  type: Phaser.AUTO,
  parent: document.body,         // attach to <body>
  backgroundColor: 0x000000,     // Black background
  width: GAME_WIDTH,
  height: GAME_HEIGHT,
  physics: {
    default: "matter",
    matter: {
      gravity: { x: 0,y: 1 },
    },
  },
  scene: [MainScene],
  render: {
    transparent: false,          // Disable transparency to prevent white screen issues
  },
  scale: {
    mode: Phaser.Scale.RESIZE,   // Resize to fit window
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: GAME_WIDTH,
    height: GAME_HEIGHT
  }
});
