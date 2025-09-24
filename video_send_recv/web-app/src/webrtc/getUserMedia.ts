import { VideoConstraints } from '../types';

export const DEFAULT_VIDEO_CONSTRAINTS: VideoConstraints = {
  width: { ideal: 640, max: 640 },
  height: { ideal: 360, max: 360 },
  frameRate: { ideal: 30, max: 30 },
  facingMode: 'user'
};

export async function getUserMedia(): Promise<MediaStream> {
  try {
    console.log('Requesting camera with constraints:', DEFAULT_VIDEO_CONSTRAINTS);
    const stream = await navigator.mediaDevices.getUserMedia({
      video: DEFAULT_VIDEO_CONSTRAINTS,
      audio: false
    });

    console.log('Camera stream obtained:', stream);
    console.log('Video tracks:', stream.getVideoTracks());
    console.log('Audio tracks:', stream.getAudioTracks());

    // Ensure frame rate constraint is applied
    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      console.log('Video track settings:', videoTrack.getSettings());
      await videoTrack.applyConstraints({ frameRate: 30 });
      console.log('Applied frame rate constraint');
    }

    return stream;
  } catch (error) {
    console.error('Error accessing camera:', error);
    throw new Error(`Camera access failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export function stopUserMedia(stream: MediaStream): void {
  stream.getTracks().forEach(track => track.stop());
}
