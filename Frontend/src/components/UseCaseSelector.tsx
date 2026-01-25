import { CheckCircle2, Trash2, Package, Eye, Users, Shield } from 'lucide-react';

export interface UseCase {
  id: string;
  name: string;
  description: string;
  icon: 'bottle' | 'trash' | 'motion' | 'people' | 'security' | 'quality';
  prompt: string;
}

interface UseCaseSelectorProps {
  selectedUseCase: UseCase | null;
  onSelect: (useCase: UseCase) => void;
  mode?: 'live' | 'demo'; // Add mode prop
}

const useCases: UseCase[] = [
  {
    id: 'restricted-area',
    name: 'Security',
    description: 'Detects unauthorized access to restricted areas using motion detection for security monitoring',
    icon: 'motion',
    prompt: 'Monitor for unauthorized access to restricted areas. Detect any person or movement that indicates potential security breach.',
  },
  {
    id: 'trash-status',
    name: 'Trash',
    description: 'Monitors trash bin status including overflow conditions, fill levels, and maintenance requirements',
    icon: 'trash',
    prompt: 'Analyze the trash bin status in the video. Report if bins are overflowing, need emptying, or show maintenance issues.',
  },
  {
    id: 'production-anomaly',
    name: 'Production',
    description: 'Detects production line anomalies including bottle positioning, flow issues, and equipment malfunctions',
    icon: 'bottle',
    prompt: 'Monitor the production line for anomalies. The frames show filled bottles moving on a conveyor belt from left to right. There is a metallic arm separator in the middle - bottles should normally be on the left side (upper part). Report if any bottles are pushed to the right side (bottom part) of the separator, which indicates an issue.',
  }
];

const getIconComponent = (iconType: UseCase['icon']) => {
  const iconClass = "w-6 h-6";
  
  switch (iconType) {
    case 'bottle':
      return <Package className={iconClass} />;
    case 'trash':
      return <Trash2 className={iconClass} />;
    case 'motion':
      return <Eye className={iconClass} />;
    case 'people':
      return <Users className={iconClass} />;
    case 'security':
      return <Shield className={iconClass} />;
    case 'quality':
      return <CheckCircle2 className={iconClass} />;
    default:
      return <Eye className={iconClass} />;
  }
};

export default function UseCaseSelector({ selectedUseCase, onSelect, mode = 'demo' }: UseCaseSelectorProps) {
  // Filter use cases based on mode
  const availableUseCases = mode === 'live' 
    ? useCases.filter(useCase => useCase.id === 'restricted-area')
    : useCases.filter(useCase => useCase.id !== 'restricted-area');

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-lg">
      <h3 className="text-xl font-semibold text-blue-800 mb-6">
        {mode === 'live' ? 'Security Monitoring' : 'Analysis Use Cases'}
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {availableUseCases.map((useCase) => (
          <button
            key={useCase.id}
            onClick={() => onSelect(useCase)}
            className={`
              relative p-5 rounded-xl text-left transition-all duration-200 group shadow-sm
              ${selectedUseCase?.id === useCase.id
                ? 'bg-blue-600 text-white border-2 border-blue-700 shadow-lg transform scale-[1.02]'
                : 'bg-white hover:bg-blue-50 text-gray-800 border-2 border-gray-200 hover:border-blue-300 hover:shadow-md'
              }
            `}
          >
            <div className="flex flex-col items-center text-center gap-4">
              <div className={`
                p-4 rounded-xl flex-shrink-0
                ${selectedUseCase?.id === useCase.id
                  ? 'bg-blue-700 text-white'
                  : 'bg-blue-100 text-blue-600 group-hover:bg-blue-200'
                }
              `}>
                {getIconComponent(useCase.icon)}
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-lg">
                  {useCase.name}
                </h4>
              </div>
            </div>
          </button>
        ))}
      </div>

      {selectedUseCase && (
        <div className="border-t border-gray-200 pt-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">
            Selected: {selectedUseCase.name}
          </h4>
          <p className="text-base text-gray-600 leading-relaxed">
            {selectedUseCase.description}
          </p>
        </div>
      )}
    </div>
  );
}
