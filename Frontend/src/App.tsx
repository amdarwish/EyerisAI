import { useState, useEffect } from 'react';
import Header from './components/Header';
import VideoDisplay from './components/VideoDisplay';
import UseCaseSelector, { UseCase } from './components/UseCaseSelector';
import ControlPanel from './components/ControlPanel';
import ResultsPanel, { Detection } from './components/ResultsPanel';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SystemStatus {
  status: string;
  config: {
    ai_model: string;
    motion_detection_active: boolean;
    email_enabled: boolean;
    tts_enabled: boolean;
  };
  camera_available: boolean;
}

function App() {
  const [status, setStatus] = useState<'idle' | 'processing' | 'alert'>('idle');
  const [mode, setMode] = useState<'live' | 'demo'>('demo');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [selectedUseCase, setSelectedUseCase] = useState<UseCase | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [sessionStats, setSessionStats] = useState({
    totalAnalyses: 0,
    totalAlerts: 0,
    totalNormal: 0,
    sessionStartTime: new Date()
  });

  const canStart =
    (mode === 'demo' ? videoFile !== null : true) && selectedUseCase !== null;

  // Initialize WebSocket for real-time updates
  useEffect(() => {
    if (mode === 'live') {
      // Auto-select security use case for live stream
      const securityUseCase = {
        id: 'restricted-area',
        name: 'Security',
        description: 'Detects unauthorized access to restricted areas using motion detection for security monitoring',
        icon: 'motion' as const,
        prompt: 'Monitor for unauthorized access to restricted areas. Detect any person or movement that indicates potential security breach.',
      };
      setSelectedUseCase(securityUseCase);
      
      const ws = new WebSocket(`ws://localhost:8000/api/ws/motion`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnection(ws);
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'detection') {
          const newDetection: Detection = {
            frame: 0, // Will be set by backend
            timestamp: new Date(data.data.timestamp).toLocaleTimeString(),
            status: data.data.alert_sent ? 'alert' : 'normal',
            confidence: 95, // Default confidence
            description: data.data.description
          };
          
          setDetections(prev => [...prev, newDetection]);
          
          // Update session statistics
          setSessionStats(prev => ({
            ...prev,
            totalAnalyses: prev.totalAnalyses + 1,
            totalAlerts: data.data.alert_sent ? prev.totalAlerts + 1 : prev.totalAlerts,
            totalNormal: !data.data.alert_sent ? prev.totalNormal + 1 : prev.totalNormal
          }));
          
          if (data.data.alert_sent) {
            setStatus('alert');
          }
        } else if (data.type === 'status_update') {
          if (data.data.motion_detection === 'started') {
            setStatus('processing');
          } else if (data.data.motion_detection === 'stopped') {
            setStatus('idle');
          }
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnection(null);
      };
      
      return () => {
        ws.close();
      };
    } else {
      // Reset use case when switching to video upload mode
      setSelectedUseCase(null);
    }
  }, [mode]);

  // Load system status on mount
  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  const handleStart = async () => {
    setIsProcessing(true);
    setStatus('processing');

    try {
      if (mode === 'live') {
        // Start live motion detection with use case specific prompt
        if (selectedUseCase?.id === 'restricted-area') {
          console.log('\n' + '='.repeat(60));
          console.log('🚀 STARTING LIVE MOTION DETECTION');
          console.log('='.repeat(60));
          console.log('📋 Terminal Command Equivalent:');
          console.log('   python backend/EyerisAI.py --use-case motion');
          console.log('📍 API Endpoint: POST /motion/start');
          console.log('🎯 Use Case: Security - Restricted Area Access');
          console.log('='.repeat(60));
          console.log('📤 Sending request to backend...');
          
          const response = await fetch(`${API_BASE_URL}/motion/start`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          
          if (!response.ok) {
            throw new Error(`Failed to start motion detection: ${response.statusText}`);
          }
          
          const result = await response.json();
          console.log('\n' + '='.repeat(60));
          console.log('✅ MOTION DETECTION STARTED SUCCESSFULLY');
          console.log('='.repeat(60));
          console.log('📊 Result:', result);
          console.log('🔴 Status: Live monitoring active');
          console.log('📧 Email alerts: Enabled for security breaches');
          console.log('='.repeat(60) + '\n');
        } else {
          throw new Error('Live mode is only available for Restricted Area Access monitoring');
        }
        
      } else {
        // Process uploaded video for trash status or production anomaly
        if (!videoFile || !selectedUseCase) {
          throw new Error('Video file and use case are required');
        }
        
        // Log the backend command that would be equivalent
        const nFrames = selectedUseCase.id === 'trash-status' ? '3' : '4';
        const interval = selectedUseCase.id === 'trash-status' ? '5.0' : 
                        selectedUseCase.id === 'production-anomaly' ? '2.0' : '10.0';
        
        console.log('\n' + '='.repeat(60));
        console.log('🎥 STARTING VIDEO ANALYSIS');
        console.log('='.repeat(60));
        console.log('📋 Terminal Command Equivalent:');
        console.log(`   python backend/EyerisAI.py --use-case video --video "${videoFile.name}" --n-frames ${nFrames} --interval ${interval}`);
        console.log('📍 API Endpoint: POST /video/analyze');
        console.log(`🎯 Use Case: ${selectedUseCase.id === 'trash-status' ? 'Trash Monitoring' : 'Production Line Monitoring'}`);
        console.log(`📊 Parameters: n_frames=${nFrames}, interval=${interval}`);
        console.log('='.repeat(60));
        console.log('📤 Sending video analysis request to backend...');
        
        const formData = new FormData();
        formData.append('file', videoFile);
        formData.append('use_case', selectedUseCase.id === 'trash-status' ? 'trash' : 'bottles');
        
        // Set appropriate parameters based on use case
        if (selectedUseCase.id === 'trash-status') {
          formData.append('n_frames', '3');
          formData.append('interval', '5.0');
        } else if (selectedUseCase.id === 'production-anomaly') {
          formData.append('n_frames', '4');
          formData.append('interval', '2.0');
        } else {
          formData.append('n_frames', '4');
          formData.append('interval', '10.0');
        }
        
        const response = await fetch(`${API_BASE_URL}/video/analyze`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error(`Video analysis failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('\n' + '='.repeat(60));
        console.log('✅ VIDEO ANALYSIS COMPLETED SUCCESSFULLY');
        console.log('='.repeat(60));
        console.log('📊 Analysis Result:', result.analysis);
        console.log('🔍 Frames Analyzed:', result.frames_analyzed);
        console.log('⚠️  Alert Status:', result.alert_sent ? 'ALERT TRIGGERED' : 'Normal - No Issues');
        if (result.alert_sent) {
          console.log('📧 Email Alert: Sent to configured recipients');
        }
        console.log('='.repeat(60) + '\n');
        
        // Convert API results to Detection format
        const newDetection: Detection = {
          frame: result.frames_analyzed,
          timestamp: new Date(result.timestamp).toLocaleTimeString(),
          status: result.alert_sent ? 'alert' : 'normal',
          confidence: 95, // Default confidence
          description: result.analysis
        };
        
        setDetections(prev => [...prev, newDetection]);
        
        // Update session statistics
        setSessionStats(prev => ({
          ...prev,
          totalAnalyses: prev.totalAnalyses + 1,
          totalAlerts: result.alert_sent ? prev.totalAlerts + 1 : prev.totalAlerts,
          totalNormal: !result.alert_sent ? prev.totalNormal + 1 : prev.totalNormal
        }));
        
        // Check for alerts
        setStatus(result.alert_sent ? 'alert' : 'idle');
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error starting analysis:', error);
      setStatus('idle');
      setIsProcessing(false);
      alert(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleStop = async () => {
    try {
      if (mode === 'live') {
        const response = await fetch(`${API_BASE_URL}/motion/stop`, {
          method: 'POST',
        });
        
        if (!response.ok) {
          throw new Error(`Failed to stop motion detection: ${response.statusText}`);
        }
      }
    } catch (error) {
      console.error('Error stopping analysis:', error);
    }
    
    setIsProcessing(false);
    setStatus('idle');
  };

  const handleReset = () => {
    setStatus('idle');
    setVideoFile(null);
    setDetections([]);
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-100">
      <Header />

      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <VideoDisplay
              mode={mode}
              onModeChange={setMode}
              videoFile={videoFile}
              onVideoFileChange={setVideoFile}
              isProcessing={isProcessing}
              selectedUseCase={selectedUseCase?.id || null}
            />

            <ControlPanel
              isProcessing={isProcessing}
              canStart={canStart}
              onStart={handleStart}
              onStop={handleStop}
              onReset={handleReset}
            />
          </div>

          <div className="space-y-6">
            <UseCaseSelector
              selectedUseCase={selectedUseCase}
              onSelect={setSelectedUseCase}
              mode={mode}
            />
            <ResultsPanel 
              detections={detections} 
              isProcessing={isProcessing}
              sessionStats={sessionStats}
              onClearResults={() => {
                setDetections([]);
                setSessionStats({
                  totalAnalyses: 0,
                  totalAlerts: 0,
                  totalNormal: 0,
                  sessionStartTime: new Date()
                });
              }}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
