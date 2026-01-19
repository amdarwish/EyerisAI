import { AlertTriangle, CheckCircle, Clock, TrendingUp, RotateCcw, Calendar, Timer } from 'lucide-react';

export interface Detection {
  frame: number;
  timestamp: string;
  status: 'normal' | 'alert';
  confidence: number;
  description: string;
}

interface SessionStats {
  totalAnalyses: number;
  totalAlerts: number;
  totalNormal: number;
  sessionStartTime: Date;
}

interface ResultsPanelProps {
  detections: Detection[];
  isProcessing: boolean;
  sessionStats?: SessionStats;
  onClearResults?: () => void;
}

export default function ResultsPanel({
  detections,
  isProcessing,
  sessionStats,
  onClearResults,
}: ResultsPanelProps) {
  const hasAlerts = detections.some((d) => d.status === 'alert');
  const alertCount = detections.filter((d) => d.status === 'alert').length;
  const normalCount = detections.filter((d) => d.status === 'normal').length;
  
  // Calculate session duration
  const sessionDuration = sessionStats 
    ? Math.round((Date.now() - sessionStats.sessionStartTime.getTime()) / 1000 / 60) 
    : 0;

  return (
    <div className="space-y-4">
      {hasAlerts && (
        <div className="bg-status-alert/10 border border-status-alert rounded-xl p-4 animate-pulse">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-status-alert flex-shrink-0" />
            <div>
              <h3 className="text-status-alert font-semibold">
                Critical Anomaly Detected
              </h3>
              <p className="text-status-alert/80 text-sm">
                {alertCount} alert{alertCount !== 1 ? 's' : ''} require immediate
                attention
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-blue-800">Analysis Results</h2>
          {onClearResults && (
            <button
              onClick={onClearResults}
              disabled={!sessionStats || sessionStats.totalAnalyses === 0}
              className={`flex items-center gap-2 px-3 py-1 text-sm rounded-lg transition-colors ${
                sessionStats && sessionStats.totalAnalyses > 0
                  ? 'text-red-600 hover:text-red-700 hover:bg-red-50'
                  : 'text-gray-400 cursor-not-allowed'
              }`}
              title={sessionStats && sessionStats.totalAnalyses > 0 ? "Reset session statistics" : "No data to clear"}
            >
              <RotateCcw className="w-4 h-4" />
              Reset Session
            </button>
          )}
        </div>

        {/* 3-Box Layout */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="bg-blue-50 rounded-lg p-4 text-center border border-blue-200">
            <div className="text-2xl font-bold text-blue-800 mb-1">
              {sessionStats?.totalAnalyses || 0}
            </div>
            <div className="text-sm text-blue-600">Total</div>
          </div>

          <div className="bg-green-50 rounded-lg p-4 text-center border border-green-200">
            <div className="text-2xl font-bold text-green-800 mb-1">
              {sessionStats?.totalNormal || 0}
            </div>
            <div className="text-sm text-green-600">Normal</div>
          </div>

          <div className="bg-red-50 rounded-lg p-4 text-center border border-red-200">
            <div className="text-2xl font-bold text-red-800 mb-1">
              {sessionStats?.totalAlerts || 0}
            </div>
            <div className="text-sm text-red-600">Alert</div>
          </div>
        </div>

        {/* Session Info */}
        {!sessionStats && (
          <div className="text-center py-8">
            <Clock className="w-8 h-8 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-600 text-sm">No session data available</p>
          </div>
        )}
      </div>
    </div>
  );
}
