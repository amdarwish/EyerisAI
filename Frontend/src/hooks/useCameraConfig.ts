import { useState, useEffect } from 'react';
import cameraConfig from '../config/camera.json';

interface CameraConfig {
  camera: {
    enabled: boolean;
    ip_url: string;
    fallback_message: string;
    auto_connect: boolean;
    refresh_interval: number;
  };
  display: {
    show_controls: boolean;
    aspect_ratio: string;
    max_width: string;
  };
}

export const useCameraConfig = () => {
  const [config] = useState<CameraConfig>(cameraConfig);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testConnection = async (url: string): Promise<boolean> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        setIsConnected(true);
        setError(null);
        resolve(true);
      };
      img.onerror = () => {
        setIsConnected(false);
        setError('Camera not accessible');
        resolve(false);
      };
      img.src = url;
      
      // Timeout after 5 seconds
      setTimeout(() => {
        setIsConnected(false);
        setError('Connection timeout');
        resolve(false);
      }, 5000);
    });
  };

  useEffect(() => {
    if (config.camera.enabled && config.camera.auto_connect) {
      testConnection(config.camera.ip_url);
    }
  }, [config.camera.enabled, config.camera.auto_connect, config.camera.ip_url]);

  return {
    config,
    isConnected,
    error,
    testConnection,
  };
};