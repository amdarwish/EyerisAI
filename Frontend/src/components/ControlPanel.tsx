import { Play, Square, RotateCcw } from 'lucide-react';

interface ControlPanelProps {
  isProcessing: boolean;
  canStart: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

export default function ControlPanel({
  isProcessing,
  canStart,
  onStart,
  onStop,
  onReset,
}: ControlPanelProps) {
  return (
    <div>
      <div className="grid grid-cols-3 gap-3">
        <button
          onClick={onStart}
          disabled={!canStart || isProcessing}
          className={`flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-semibold transition-all duration-200 ${
            !canStart || isProcessing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed border-2 border-gray-300'
              : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-xl transform hover:scale-[1.02] border-2 border-green-600'
          }`}
        >
          <Play className="w-5 h-5" />
          Start Analysis
        </button>

        <button
          onClick={onStop}
          disabled={!isProcessing}
          className={`flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-semibold transition-all duration-200 ${
            !isProcessing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed border-2 border-gray-300'
              : 'bg-red-600 hover:bg-red-700 text-white shadow-lg hover:shadow-xl transform hover:scale-[1.02] border-2 border-red-600'
          }`}
        >
          <Square className="w-5 h-5" />
          Stop
        </button>

        <button
          onClick={onReset}
          disabled={isProcessing}
          className={`flex items-center justify-center gap-2 px-6 py-4 rounded-lg font-semibold transition-all ${
            isProcessing
              ? 'bg-primary-700 text-primary-500 cursor-not-allowed'
              : 'bg-primary-700 hover:bg-primary-600 text-white'
          }`}
        >
          <RotateCcw className="w-5 h-5" />
          Reset
        </button>
      </div>
    </div>
  );
}
