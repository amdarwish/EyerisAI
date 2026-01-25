import { Video, Camera, Upload, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useCameraConfig } from '../hooks/useCameraConfig';

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
  selectedUseCase,
}: VideoDisplayProps) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const { config, isConnected, error } = useCameraConfig();

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

      <div className="aspect-video bg-gradient-to-br from-gray-50 to-gray-100 relative flex items-center justify-center">
        {mode === 'live' ? (
          <div className="w-full h-full flex items-center justify-center">
            {config.camera.enabled && config.camera.ip_url ? (
              isConnected ? (
                <img
                  src={config.camera.ip_url}
                  alt="Live Camera Stream"
                  className="w-full h-full object-contain"
                  style={{ maxWidth: config.display.max_width }}
                />
              ) : (
                <div className="text-center p-8">
                  <Camera className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-800 mb-2">Live Streaming on link below</p>
                  <p className="text-xs text-gray-500">{config.camera.ip_url}</p>
                </div>
              )
            ) : (
              <div className="text-center p-8">
                <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-800 mb-2">Camera Not Configured</p>
                <p className="text-sm text-gray-600 mb-4">{config.camera.fallback_message}</p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-left">
                  <p className="text-sm font-medium text-blue-800 mb-2">📝 Configuration:</p>
                  <p className="text-xs text-blue-700 font-mono">Frontend/src/config/camera.json</p>
                  <p className="text-xs text-blue-600 mt-1">Update the ip_url field with your camera stream</p>
                </div>
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
