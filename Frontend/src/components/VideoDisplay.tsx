import { Video, Camera, Upload, X, VideoOff, Laptop, Globe, Settings } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';

type CameraSource = 'builtin' | 'external';

interface VideoDisplayProps {
  mode: 'live' | 'demo';
  onModeChange: (mode: 'live' | 'demo') => void;
  videoFile: File | null;
  onVideoFileChange: (file: File | null) => void;
  isProcessing: boolean;
  selectedUseCase?: string | null;
}

export default function VideoDisplay({
  mode,
  onModeChange,
  videoFile,
  onVideoFileChange,
  isProcessing,
}: VideoDisplayProps) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [cameraSource, setCameraSource] = useState<CameraSource>('builtin');
  const [externalUrl, setExternalUrl] = useState<string>('');
  const [showCameraSettings, setShowCameraSettings] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startWebcam = async () => {
    try {
      setWebcamError(null);
      setWebcamActive(false);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      streamRef.current = stream;

      // Wait a tick for the video element to be in the DOM
      setTimeout(() => {
        if (videoRef.current && streamRef.current) {
          videoRef.current.srcObject = streamRef.current;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setWebcamActive(true);
          };
        }
      }, 100);
    } catch (err) {
      console.error('Webcam error:', err);
      setWebcamError(
        err instanceof Error
          ? err.message
          : 'Failed to access webcam. Please allow camera permissions.'
      );
      setWebcamActive(false);
    }
  };

  const connectExternalCamera = () => {
    if (!externalUrl.trim()) {
      setWebcamError('Please enter a valid camera URL');
      return;
    }
    setWebcamError(null);
    setWebcamActive(true);
  };

  // Handle camera for live mode
  useEffect(() => {
    if (mode === 'live') {
      if (cameraSource === 'builtin') {
        // Small delay to ensure component is mounted
        const timer = setTimeout(() => {
          startWebcam();
        }, 200);
        return () => clearTimeout(timer);
      } else {
        // External camera - stop any webcam stream
        stopWebcam();
        if (externalUrl.trim()) {
          setWebcamActive(true);
        }
      }
    } else {
      stopWebcam();
      setWebcamActive(false);
    }

    return () => {
      if (cameraSource === 'builtin') {
        stopWebcam();
      }
    };
  }, [mode, cameraSource]);

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
  };

  // Create video URL when file is selected
  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setVideoUrl(null);
    }
  }, [videoFile]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    onVideoFileChange(file);
  };

  const handleRemoveVideo = () => {
    onVideoFileChange(null);
  };

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 overflow-hidden shadow-lg">
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 px-6 py-4 border-b-2 border-blue-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-blue-800 flex items-center gap-2">
            <Video className="w-5 h-5 text-blue-600" />
            Video Input
          </h2>

          <div className="flex gap-2 bg-white border-2 border-blue-200 rounded-xl p-1 shadow-sm">
            <button
              onClick={() => onModeChange('live')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                mode === 'live'
                  ? 'bg-accent-blue text-white'
                  : 'text-primary-300 hover:text-black'
              }`}
            >
              <Camera className="w-4 h-4 inline-block mr-2" />
              Live Stream
            </button>
            <button
              onClick={() => onModeChange('demo')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                mode === 'demo'
                  ? 'bg-accent-blue text-white'
                  : 'text-primary-300 hover:text-black'
              }`}
            >
              <Upload className="w-4 h-4 inline-block mr-2" />
              Video Upload
            </button>
          </div>
        </div>
      </div>

      {/* Camera Settings Panel for Live Mode */}
      {mode === 'live' && (
        <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center gap-4 flex-wrap">
            <span className="text-sm font-medium text-gray-700">Camera Source:</span>
            <div className="flex gap-2">
              <button
                onClick={() => setCameraSource('builtin')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  cameraSource === 'builtin'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Laptop className="w-4 h-4" />
                Built-in Webcam
              </button>
              <button
                onClick={() => setCameraSource('external')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  cameraSource === 'external'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Globe className="w-4 h-4" />
                External IP Camera
              </button>
            </div>

            {cameraSource === 'external' && (
              <div className="flex items-center gap-2 flex-1 min-w-[300px]">
                <input
                  type="text"
                  value={externalUrl}
                  onChange={(e) => setExternalUrl(e.target.value)}
                  placeholder="http://192.168.1.100:8080/video"
                  className="flex-1 px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  onClick={connectExternalCamera}
                  className="px-4 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Connect
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="aspect-video bg-gradient-to-br from-gray-900 to-gray-800 relative flex items-center justify-center">
        {mode === 'live' ? (
          <div className="w-full h-full relative">
            {/* Built-in webcam video element */}
            {cameraSource === 'builtin' && (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-contain ${webcamActive ? 'block' : 'hidden'}`}
              />
            )}

            {/* External IP camera image/stream */}
            {cameraSource === 'external' && webcamActive && externalUrl && (
              <img
                src={externalUrl}
                alt="External Camera Stream"
                className="w-full h-full object-contain"
                onError={() => setWebcamError('Failed to load external camera stream. Check the URL.')}
              />
            )}

            {/* Error state */}
            {webcamError && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-80">
                <div className="text-center p-8">
                  <VideoOff className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-white mb-2">Camera Error</p>
                  <p className="text-sm text-gray-300 mb-4">{webcamError}</p>
                  <button
                    onClick={() => cameraSource === 'builtin' ? startWebcam() : connectExternalCamera()}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            )}

            {/* Loading state */}
            {!webcamActive && !webcamError && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center p-8">
                  <Camera className="w-16 h-16 text-blue-400 mx-auto mb-4 animate-pulse" />
                  <p className="text-lg font-medium text-white mb-2">
                    {cameraSource === 'builtin' ? 'Connecting to Webcam...' : 'Enter Camera URL Above'}
                  </p>
                  <p className="text-sm text-gray-400">
                    {cameraSource === 'builtin'
                      ? 'Please allow camera access when prompted'
                      : 'Enter the IP camera URL and click Connect'}
                  </p>
                </div>
              </div>
            )}

            {/* Live indicator */}
            {webcamActive && !webcamError && (
              <div className="absolute top-4 left-4 flex items-center gap-2 bg-black bg-opacity-60 px-3 py-1.5 rounded-lg">
                <span className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse"></span>
                <span className="text-white text-sm font-medium">LIVE</span>
                <span className="text-gray-400 text-xs ml-2">
                  {cameraSource === 'builtin' ? 'Webcam' : 'IP Camera'}
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="w-full h-full flex items-center justify-center p-8">
            {videoFile && videoUrl ? (
              <div className="w-full h-full relative">
                <video
                  src={videoUrl}
                  controls
                  className="w-full h-full object-contain rounded-lg"
                  style={{ maxHeight: '100%' }}
                >
                  Your browser does not support the video tag.
                </video>
                
                {/* Video info overlay */}
                <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white px-3 py-2 rounded-lg text-sm">
                  <p className="font-medium">{videoFile.name}</p>
                  <p className="text-gray-300">
                    {(videoFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>

                {/* Remove video button */}
                <button
                  onClick={handleRemoveVideo}
                  className="absolute top-4 right-4 bg-red-500 hover:bg-red-600 text-white p-2 rounded-lg transition-colors"
                  title="Remove video"
                >
                  <X className="w-4 h-4" />
                </button>

                {isProcessing && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                    <div className="bg-white rounded-lg p-6 text-center">
                      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden mb-3">
                        <div className="h-full bg-blue-500 animate-pulse"></div>
                      </div>
                      <p className="text-blue-600 font-medium">Processing video...</p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <label className="cursor-pointer group w-full">
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="border-2 border-dashed border-gray-300 group-hover:border-blue-400 rounded-xl p-12 text-center transition-colors">
                  <Upload className="w-16 h-16 text-gray-400 group-hover:text-blue-500 mx-auto mb-4 transition-colors" />
                  <p className="text-gray-600 text-lg group-hover:text-gray-800 transition-colors">
                    Click to upload video file
                  </p>
                  <p className="text-gray-500 text-sm mt-2">
                    Supports MP4, AVI, MOV formats
                  </p>
                </div>
              </label>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
