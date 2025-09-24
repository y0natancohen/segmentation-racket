// Codec preferences for WebRTC video transmission
// Prefer H.264 when available, fallback to VP8/VP9

export const H264_CODEC_PREFERENCES = [
  'video/H264; profile-level-asymmetry-allowed=1; packetization-mode=1; level-asymmetry-allowed=1',
  'video/H264; profile-level-asymmetry-allowed=1; packetization-mode=0; level-asymmetry-allowed=1',
  'video/H264; profile-level-asymmetry-allowed=0; packetization-mode=1; level-asymmetry-allowed=0',
  'video/H264; profile-level-asymmetry-allowed=0; packetization-mode=0; level-asymmetry-allowed=0'
];

export const VP8_CODEC_PREFERENCES = [
  'video/VP8'
];

export const VP9_CODEC_PREFERENCES = [
  'video/VP9'
];

export async function setCodecPreferences(transceiver: RTCRtpTransceiver, codec: 'h264' | 'vp8' | 'vp9'): Promise<void> {
  try {
    const capabilities = RTCRtpSender.getCapabilities('video');
    if (!capabilities) return;

    let preferredCodecs: string[] = [];
    
    switch (codec) {
      case 'h264':
        preferredCodecs = H264_CODEC_PREFERENCES;
        break;
      case 'vp8':
        preferredCodecs = VP8_CODEC_PREFERENCES;
        break;
      case 'vp9':
        preferredCodecs = VP9_CODEC_PREFERENCES;
        break;
    }

    // Filter available codecs
    const availableCodecs = capabilities.codecs.filter(c => 
      preferredCodecs.some(pref => c.mimeType.includes(pref.split(';')[0]))
    );

    if (availableCodecs.length > 0) {
      await transceiver.setCodecPreferences(availableCodecs);
    }
  } catch (error) {
    console.warn('Failed to set codec preferences:', error);
  }
}

export function detectPreferredCodec(): 'h264' | 'vp8' | 'vp9' {
  // Simple detection based on browser capabilities
  const capabilities = RTCRtpSender.getCapabilities('video');
  if (!capabilities) return 'vp8';

  const hasH264 = capabilities.codecs.some(c => c.mimeType.includes('H264'));
  const hasVP8 = capabilities.codecs.some(c => c.mimeType.includes('VP8'));
  const hasVP9 = capabilities.codecs.some(c => c.mimeType.includes('VP9'));

  if (hasH264) return 'h264';
  if (hasVP8) return 'vp8';
  if (hasVP9) return 'vp9';
  
  return 'vp8'; // fallback
}
