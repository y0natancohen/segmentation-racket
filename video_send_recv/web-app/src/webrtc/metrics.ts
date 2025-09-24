import { IntensityMetrics, ConnectionStats } from '../types';

export class MetricsCollector {
  private latestMetrics: IntensityMetrics | null = null;
  private messageCount = 0;
  private lastMessageTime = 0;
  private messageTimes: number[] = [];
  private statsInterval: number | null = null;
  private onStatsUpdate?: (stats: ConnectionStats) => void;

  constructor() {
    this.startMessageRateCalculation();
  }

  private startMessageRateCalculation(): void {
    this.statsInterval = window.setInterval(() => {
      const now = Date.now();
      const oneSecondAgo = now - 1000;
      
      // Keep only messages from the last second
      this.messageTimes = this.messageTimes.filter(time => time > oneSecondAgo);
      
      const messagesPerSecond = this.messageTimes.length;
      
      this.onStatsUpdate?.({
        outboundFps: 0, // Will be updated by getStats()
        outboundBitrate: 0, // Will be updated by getStats()
        inboundFps: 0, // Will be updated by getStats()
        messagesPerSecond
      });
    }, 1000);
  }

  handleMessage(data: string): void {
    try {
      const metrics: IntensityMetrics = JSON.parse(data);
      this.latestMetrics = metrics;
      
      this.messageCount++;
      this.lastMessageTime = Date.now();
      this.messageTimes.push(this.lastMessageTime);
    } catch (error) {
      console.error('Failed to parse metrics message:', error);
    }
  }

  getLatestMetrics(): IntensityMetrics | null {
    return this.latestMetrics;
  }

  setOnStatsUpdate(callback: (stats: ConnectionStats) => void): void {
    this.onStatsUpdate = callback;
  }

  destroy(): void {
    if (this.statsInterval) {
      clearInterval(this.statsInterval);
      this.statsInterval = null;
    }
  }
}

export function parseRTCStats(stats: RTCStatsReport): Partial<ConnectionStats> {
  let outboundFps = 0;
  let outboundBitrate = 0;
  let inboundFps = 0;

  for (const stat of stats.values()) {
    if (stat.type === 'outbound-rtp' && stat.kind === 'video') {
      outboundFps = stat.framesPerSecond || 0;
      outboundBitrate = stat.bytesPerSecond ? (stat.bytesPerSecond * 8) : 0;
    } else if (stat.type === 'inbound-rtp' && stat.kind === 'video') {
      inboundFps = stat.framesPerSecond || 0;
    }
  }

  return {
    outboundFps,
    outboundBitrate,
    inboundFps
  };
}
