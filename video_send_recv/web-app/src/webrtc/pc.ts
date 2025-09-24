import { DataChannelConfig, SignalingRequest, SignalingResponse } from '../types';
import { setCodecPreferences, detectPreferredCodec } from './codecs';

export class WebRTCPeerConnection {
  private pc: RTCPeerConnection;
  private dataChannel: RTCDataChannel | null = null;
  private onDataChannelMessage?: (data: string) => void;
  private onConnectionStateChange?: (state: RTCPeerConnectionState) => void;

  constructor() {
    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }],
      bundlePolicy: 'max-bundle',
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.pc.onconnectionstatechange = () => {
      this.onConnectionStateChange?.(this.pc.connectionState);
    };

    this.pc.ondatachannel = (event) => {
      const channel = event.channel;
      channel.onmessage = (event) => {
        this.onDataChannelMessage?.(event.data);
      };
    };
  }

  async addVideoTrack(stream: MediaStream): Promise<void> {
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack) throw new Error('No video track found');

    const transceiver = this.pc.addTransceiver(videoTrack, { direction: 'sendonly' });
    
    // Set codec preferences
    const preferredCodec = detectPreferredCodec();
    await setCodecPreferences(transceiver, preferredCodec);

    // Configure encoding parameters
    const sender = transceiver.sender;
    const params = sender.getParameters();
    params.encodings = [{
      maxBitrate: 600_000,
      maxFramerate: 30,
      scaleResolutionDownBy: 1
    }];
    await sender.setParameters(params);
  }

  createDataChannel(config: DataChannelConfig): RTCDataChannel {
    this.dataChannel = this.pc.createDataChannel('metrics', config);
    
    this.dataChannel.onmessage = (event) => {
      this.onDataChannelMessage?.(event.data);
    };

    return this.dataChannel;
  }

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    return offer;
  }

  async handleAnswer(answer: RTCSessionDescriptionInit): Promise<void> {
    await this.pc.setRemoteDescription(answer);
  }

  async signalToServer(offer: RTCSessionDescriptionInit, serverUrl: string): Promise<void> {
    const request: SignalingRequest = {
      sdp: offer.sdp!,
      type: 'offer'
    };

    const response = await fetch(`${serverUrl}/offer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Signaling failed: ${response.status} ${response.statusText}`);
    }

    const answer: SignalingResponse = await response.json();
    await this.handleAnswer(answer);
  }

  setOnDataChannelMessage(callback: (data: string) => void): void {
    this.onDataChannelMessage = callback;
  }

  setOnConnectionStateChange(callback: (state: RTCPeerConnectionState) => void): void {
    this.onConnectionStateChange = callback;
  }

  async getStats(): Promise<RTCStatsReport> {
    return await this.pc.getStats();
  }

  close(): void {
    this.dataChannel?.close();
    this.pc.close();
  }

  get connectionState(): RTCPeerConnectionState {
    return this.pc.connectionState;
  }

  get dataChannelState(): RTCDataChannelState | null {
    return this.dataChannel?.readyState || null;
  }
}
