import Phaser from "phaser";

interface RectangleMessage {
  position: { x: number; y: number };
}

export class MainScene extends Phaser.Scene {
  private balls: Phaser.Physics.Matter.Image[] = [];
  // @ts-ignore - Used for timer callback side effects
  private ballSpawnTimer!: Phaser.Time.TimerEvent;
  private platform!: Phaser.Physics.Matter.Image;
  
  // WebSocket connection for rectangle data
  private ws: WebSocket | null = null;
  private connectionRetryTimer: number = 0;
  private connectionRetryDelay: number = 2000; // 2 seconds
  
  // Latest message storage
  private latestMessage: RectangleMessage | null = null;
  
  // FPS monitoring
  private fpsText!: Phaser.GameObjects.Text;
  private statusText!: Phaser.GameObjects.Text;
  private messageCount: number = 0;
  private lastFpsUpdateTime: number = 0;
  private currentFps: number = 0;
  private connectionStartTime: number = 0;

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
    const size = 600;
    this.cameras.main.setBackgroundColor(0xffffff);
    this.scale.resize(size, size); // 600x600 “square board”

    // Enable Matter physics
    this.matter.world.setBounds(0, 0, size, size);
    this.matter.world.setGravity(0, 1); // gravity down
    
    // Raise solver iterations for more accurate, less "mushy" collisions
    this.matter.world.engine.positionIterations = 8; // default ~6
    this.matter.world.engine.velocityIterations = 6; // default ~4
    this.matter.world.engine.constraintIterations = 4; // default ~2

    // Create initial ball
    this.createBall(size / 2, 50);

    // Set up timer to spawn new balls every second
    this.ballSpawnTimer = this.time.addEvent({
      delay: 1000, // 1 second
      callback: () => {
        // Spawn ball at random x position near the top
        const x = Phaser.Math.Between(50, size - 50);
        this.createBall(x, 50);
      },
      loop: true
    });

    // Floor (bottom of the square) with bouncy physics
    this.matter.add.rectangle(size / 2, size - 10, size - 20, 20, { 
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

    // Floating black rectangle that moves up and down
    this.platform = this.matter.add.image(size / 2, size * 0.7, "plat");
    this.platform.setStatic(true);
    // Platform (static) but still allowed to have restitution/friction:
    this.platform.setBounce(0.95);
    this.platform.setFriction(0.001);
    this.platform.setFrictionStatic(0.001);

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
      if (ball.y > 650) { // If ball is below the screen
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
      console.log("Connecting to rectangle generator WebSocket...");
      this.ws = new WebSocket("ws://localhost:8765");
      
      this.ws.onopen = () => {
        console.log("Connected to rectangle generator");
        this.connectionRetryTimer = 0; // Reset retry timer
        this.connectionStartTime = this.time.now;
        this.updateStatusDisplay();
      };
      
      this.ws.onmessage = (event) => {
        try {
          const message: RectangleMessage = JSON.parse(event.data);
          this.latestMessage = message; // Store latest message directly
          this.messageCount++; // Count received messages
        } catch (error) {
          console.error("Failed to parse rectangle message:", error);
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
      // Use latest position directly
      this.platform.setPosition(this.latestMessage.position.x, this.latestMessage.position.y);
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

  shutdown(): void {
    // Clean up WebSocket connection
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

new Phaser.Game({
  type: Phaser.AUTO,
  parent: document.body,         // attach to <body>
  backgroundColor: "#ffffff",
  width: 600,
  height: 600,
  physics: {
    default: "matter",
    matter: {
      gravity: { x: 0,y: 1 },
    },
  },
  scene: [MainScene],
});
