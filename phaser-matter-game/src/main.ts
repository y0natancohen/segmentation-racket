import Phaser from "phaser";

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
  
  // WebSocket connection for polygon data
  private ws: WebSocket | null = null;
  private connectionRetryTimer: number = 0;
  private connectionRetryDelay: number = 2000; // 2 seconds
  
  // Latest message storage
  private latestMessage: PolygonMessage | null = null;
  
  // FPS monitoring
  private fpsText!: Phaser.GameObjects.Text;
  private statusText!: Phaser.GameObjects.Text;
  private messageCount: number = 0;
  private lastFpsUpdateTime: number = 0;
  private currentFps: number = 0;
  private connectionStartTime: number = 0;
  
  // Webcam integration
  private webcamVideo!: Phaser.GameObjects.Video;
  private webcamError: string | null = null;

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

    this.initializeWebSocketConnection();

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
    this.statusText = this.add.text(10, 35, "Status: Disconnected", {
      fontSize: '14px',
      color: '#ffffff',
      backgroundColor: '#ff0000',
      padding: { x: 8, y: 4 }
    });
    this.statusText.setScrollFactor(0); // Keep text fixed on screen
    this.statusText.setDepth(1000); // Ensure it's on top

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

    this.updatePlatformFromWebSocket();
    
    this.handleConnectionRetry();
    
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

  private initializeWebSocketConnection(): void {
    try {
      console.log("Connecting to polygon generator WebSocket...");
      this.ws = new WebSocket("ws://localhost:8765");
      
      this.ws.onopen = () => {
        console.log("Connected to polygon generator");
        this.connectionRetryTimer = 0; // Reset retry timer
        this.connectionStartTime = this.time.now;
        this.updateStatusDisplay();
      };
      
      this.ws.onmessage = (event) => {
        try {
          const message: PolygonMessage = JSON.parse(event.data);
          this.latestMessage = message; // Store latest message directly
          this.messageCount++; // Count received messages
        } catch (error) {
          console.error("Failed to parse polygon message:", error);
        }
      };
      
      this.ws.onclose = () => {
        console.log("WebSocket connection closed");
        this.ws = null;
        this.connectionRetryTimer = this.time.now;
        this.updateStatusDisplay();
      };
      
      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.ws = null;
        this.connectionRetryTimer = this.time.now;
        this.updateStatusDisplay();
      };
      
    } catch (error) {
      console.error("Failed to initialize WebSocket:", error);
      this.connectionRetryTimer = this.time.now;
    }
  }

  private updatePlatformFromWebSocket(): void {
    if (this.latestMessage) {
      // Update polygon platform with full message data
      this.updatePolygonPlatform(this.latestMessage);
    }
  }

  private handleConnectionRetry(): void {
    if (!this.ws && this.connectionRetryTimer > 0) {
      const timeSinceRetry = this.time.now - this.connectionRetryTimer;
      if (timeSinceRetry >= this.connectionRetryDelay) {
        console.log("Retrying WebSocket connection...");
        this.initializeWebSocketConnection();
      }
    }
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
    if (this.ws) {
      const connectionDuration = Math.floor((this.time.now - this.connectionStartTime) / 1000);
      const hasData = this.latestMessage ? 'Yes' : 'No';
      this.statusText.setText(`Status: Connected (${connectionDuration}s) | Data: ${hasData}`);
      this.statusText.setStyle({ color: '#000000', backgroundColor: '#00ff00' });
    } else {
      this.statusText.setText("Status: Disconnected");
      this.statusText.setStyle({ color: '#ffffff', backgroundColor: '#ff0000' });
    }
  }

  private async setupWebcamBackground(): Promise<void> {
    try {
      console.log("Requesting webcam access...");
      
      // Request webcam access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: GAME_WIDTH },
          height: { ideal: GAME_HEIGHT },
          facingMode: 'user' // Front-facing camera
        } 
      });
      
      console.log("Webcam stream obtained:", stream);
      console.log("About to create video object...");
      
      // Create Phaser Video Game Object
      this.webcamVideo = this.add.video(0, 0); // Will be positioned by configureVideoLayout
      
      // Load the media stream into the Video object
      this.webcamVideo.loadMediaStream(stream);
      
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
      this.webcamError = `Webcam error: ${error instanceof Error ? error.message : 'Unknown error'}`;
      
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
    console.log('Video configured - Position:', this.webcamVideo.x, this.webcamVideo.y);
    console.log('Video configured - Size:', this.webcamVideo.displayWidth, this.webcamVideo.displayHeight);
    console.log('Video configured - Depth:', this.webcamVideo.depth);
    console.log('Game dimensions:', gameWidth, 'x', gameHeight);
    console.log('Video is playing:', this.webcamVideo.isPlaying());
    console.log('Video scale:', this.webcamVideo.scaleX, this.webcamVideo.scaleY);
    console.log('Video width/height:', this.webcamVideo.width, this.webcamVideo.height);
    
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
    // Clean up WebSocket connection
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    // Clean up webcam
    if (this.webcamVideo) {
      this.webcamVideo.stop();
      this.webcamVideo.destroy();
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
