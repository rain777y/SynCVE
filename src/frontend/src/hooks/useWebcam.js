import { useRef, useState, useCallback, useEffect } from 'react';

/**
 * Manages webcam video stream and frame capture.
 */
export function useWebcam(constraints = { width: 1280, height: 720 }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);

  const start = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: constraints.width },
          height: { ideal: constraints.height },
          facingMode: 'user',
        },
      });
      video.srcObject = stream;
      await video.play();
      setIsReady(true);
      setError(null);
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Failed to access webcam. Please check permissions.');
      setIsReady(false);
    }
  }, [constraints.width, constraints.height]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== video.HAVE_ENOUGH_DATA) {
      return null;
    }
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.95);
  }, []);

  const cleanup = useCallback(() => {
    const video = videoRef.current;
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }
    setIsReady(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup]);

  return { videoRef, canvasRef, isReady, error, start, captureFrame, cleanup };
}
