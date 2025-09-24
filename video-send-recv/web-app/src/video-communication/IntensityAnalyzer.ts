/**
 * Intensity Analyzer - Advanced intensity analysis and monitoring.
 * 
 * This module provides sophisticated intensity analysis capabilities.
 */

import { IntensityMetrics } from '../types';

export interface IntensityAlert {
  id: string;
  timestamp: number;
  connectionId: string;
  type: 'low' | 'high' | 'sudden_change' | 'stability';
  message: string;
  value: number;
  threshold?: number;
}

export interface IntensityHistory {
  connectionId: string;
  values: number[];
  timestamps: number[];
  maxSamples: number;
}

export interface IntensityStats {
  current: number;
  average: number;
  min: number;
  max: number;
  standardDeviation: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  stability: number; // 0-1, higher means more stable
}

export class IntensityAnalyzer {
  private history: Map<string, IntensityHistory> = new Map();
  private alerts: IntensityAlert[] = [];
  private config: {
    lowThreshold: number;
    highThreshold: number;
    changeThreshold: number;
    stabilityWindow: number;
    maxHistory: number;
  };

  constructor(config?: Partial<typeof IntensityAnalyzer.prototype.config>) {
    this.config = {
      lowThreshold: 10,
      highThreshold: 240,
      changeThreshold: 50,
      stabilityWindow: 10,
      maxHistory: 100,
      ...config
    };
  }

  /**
   * Process intensity data and perform analysis.
   */
  processIntensity(connectionId: string, data: IntensityMetrics): IntensityStats {
    // Update history
    this.updateHistory(connectionId, data);

    // Get current stats
    const stats = this.calculateStats(connectionId);

    // Check for alerts
    this.checkAlerts(connectionId, data, stats);

    return stats;
  }

  /**
   * Get intensity statistics for a connection.
   */
  getIntensityStats(connectionId: string): IntensityStats | null {
    const history = this.history.get(connectionId);
    if (!history || history.values.length === 0) {
      return null;
    }

    return this.calculateStats(connectionId);
  }

  /**
   * Get all alerts.
   */
  getAllAlerts(): IntensityAlert[] {
    return [...this.alerts];
  }

  /**
   * Get alerts for a specific connection.
   */
  getAlertsForConnection(connectionId: string): IntensityAlert[] {
    return this.alerts.filter(alert => alert.connectionId === connectionId);
  }

  /**
   * Clear all alerts.
   */
  clearAlerts(): void {
    this.alerts = [];
  }

  /**
   * Clear alerts for a specific connection.
   */
  clearAlertsForConnection(connectionId: string): void {
    this.alerts = this.alerts.filter(alert => alert.connectionId !== connectionId);
  }

  /**
   * Get intensity history for a connection.
   */
  getIntensityHistory(connectionId: string): IntensityHistory | null {
    return this.history.get(connectionId) || null;
  }

  /**
   * Get trend analysis for a connection.
   */
  getTrendAnalysis(connectionId: string, windowSize: number = 10): {
    trend: 'increasing' | 'decreasing' | 'stable';
    slope: number;
    correlation: number;
  } | null {
    const history = this.history.get(connectionId);
    if (!history || history.values.length < windowSize) {
      return null;
    }

    const values = history.values.slice(-windowSize);
    const timestamps = history.timestamps.slice(-windowSize);
    
    // Calculate linear regression
    const n = values.length;
    const sumX = timestamps.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = timestamps.reduce((sum, t, i) => sum + t * values[i], 0);
    const sumXX = timestamps.reduce((sum, t) => sum + t * t, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const correlation = this.calculateCorrelation(timestamps, values);
    
    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (slope > 0.1) trend = 'increasing';
    else if (slope < -0.1) trend = 'decreasing';
    
    return { trend, slope, correlation };
  }

  private updateHistory(connectionId: string, data: IntensityMetrics): void {
    if (!this.history.has(connectionId)) {
      this.history.set(connectionId, {
        connectionId,
        values: [],
        timestamps: [],
        maxSamples: this.config.maxHistory
      });
    }

    const history = this.history.get(connectionId)!;
    
    history.values.push(data.avg_intensity);
    history.timestamps.push(data.ts);
    
    // Keep only maxSamples
    if (history.values.length > this.config.maxHistory) {
      history.values.shift();
      history.timestamps.shift();
    }
  }

  private calculateStats(connectionId: string): IntensityStats {
    const history = this.history.get(connectionId);
    if (!history || history.values.length === 0) {
      throw new Error('No history available for connection');
    }

    const values = history.values;
    const current = values[values.length - 1];
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Calculate standard deviation
    const variance = values.reduce((sum, val) => sum + Math.pow(val - average, 2), 0) / values.length;
    const standardDeviation = Math.sqrt(variance);
    
    // Calculate trend
    const trend = this.calculateTrend(values);
    
    // Calculate stability (inverse of standard deviation, normalized)
    const stability = Math.max(0, Math.min(1, 1 - (standardDeviation / 100)));
    
    return {
      current,
      average,
      min,
      max,
      standardDeviation,
      trend,
      stability
    };
  }

  private calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
    if (values.length < 2) return 'stable';
    
    const recent = values.slice(-5);
    const older = values.slice(-10, -5);
    
    if (recent.length === 0 || older.length === 0) return 'stable';
    
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
    
    const change = recentAvg - olderAvg;
    
    if (change > 5) return 'increasing';
    if (change < -5) return 'decreasing';
    return 'stable';
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private checkAlerts(connectionId: string, data: IntensityMetrics, stats: IntensityStats): void {
    const timestamp = Date.now();
    
    // Low intensity alert
    if (data.avg_intensity < this.config.lowThreshold) {
      this.addAlert({
        id: `low_${timestamp}`,
        timestamp,
        connectionId,
        type: 'low',
        message: `Very low intensity: ${data.avg_intensity.toFixed(1)}`,
        value: data.avg_intensity,
        threshold: this.config.lowThreshold
      });
    }
    
    // High intensity alert
    if (data.avg_intensity > this.config.highThreshold) {
      this.addAlert({
        id: `high_${timestamp}`,
        timestamp,
        connectionId,
        type: 'high',
        message: `Very high intensity: ${data.avg_intensity.toFixed(1)}`,
        value: data.avg_intensity,
        threshold: this.config.highThreshold
      });
    }
    
    // Sudden change alert
    if (stats.standardDeviation > this.config.changeThreshold) {
      this.addAlert({
        id: `change_${timestamp}`,
        timestamp,
        connectionId,
        type: 'sudden_change',
        message: `Sudden intensity change detected`,
        value: stats.standardDeviation,
        threshold: this.config.changeThreshold
      });
    }
    
    // Stability alert
    if (stats.stability < 0.3) {
      this.addAlert({
        id: `stability_${timestamp}`,
        timestamp,
        connectionId,
        type: 'stability',
        message: `Low stability detected: ${(stats.stability * 100).toFixed(1)}%`,
        value: stats.stability
      });
    }
  }

  private addAlert(alert: IntensityAlert): void {
    this.alerts.push(alert);
    
    // Keep only last 100 alerts
    if (this.alerts.length > 100) {
      this.alerts.shift();
    }
    
    console.warn(`Intensity Alert: ${alert.message}`);
  }
}
