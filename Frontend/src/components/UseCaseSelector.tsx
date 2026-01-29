import { Package, Eye, Shield, BedDouble, HardHat, TestTube2 } from 'lucide-react';

export interface UseCase {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: 'bottle' | 'motion' | 'security' | 'bed' | 'helmet' | 'tube';
  prompt: string;
  backendValue: string; // Value to send to backend
}

interface UseCaseSelectorProps {
  selectedUseCase: UseCase | null;
  onSelect: (useCase: UseCase) => void;
  mode?: 'live' | 'demo';
}

const useCases: UseCase[] = [
  // Live Stream Use Case
  {
    id: 'restricted-area',
    name: 'Security Monitoring',
    category: 'Security',
    description: 'Detects unauthorized access to restricted areas using motion detection for real-time security monitoring',
    icon: 'motion',
    prompt: 'Monitor for unauthorized access to restricted areas. Detect any person or movement that indicates potential security breach.',
    backendValue: 'motion',
  },
  // Video Upload Use Cases
  {
    id: 'bottles',
    name: 'Bottle Detection',
    category: 'Production Line',
    description: 'Detects production line anomalies including bottle positioning, flow issues, and equipment malfunctions on conveyor belts',
    icon: 'bottle',
    prompt: 'Monitor the production line for anomalies. The frames show filled bottles moving on a conveyor belt from left to right. There is a metallic arm separator in the middle - bottles should normally be on the left side (upper part). Report if any bottles are pushed to the right side (bottom part) of the separator, which indicates an issue.',
    backendValue: 'bottles',
  },
  {
    id: 'bed',
    name: 'Patient Fall Detection',
    category: 'Healthcare',
    description: 'Monitors patients for unusual movements or falls from bed. Detects if a patient has fallen to the floor for immediate medical attention',
    icon: 'bed',
    prompt: 'Monitor the patient in the hospital bed. Detect if the patient shows any signs of falling or has fallen from the bed to the floor. Report any unusual movements or emergency situations.',
    backendValue: 'bed',
  },
  {
    id: 'helmet',
    name: 'Helmet Compliance',
    category: 'Safety Compliance',
    description: 'Checks if workers are wearing their safety helmets properly. Detects when a worker removes or is not wearing their required safety helmet',
    icon: 'helmet',
    prompt: 'Monitor workers for safety helmet compliance. Detect if any worker removes their helmet or is not wearing their safety helmet as required. Report any safety violations.',
    backendValue: 'helmet',
  },
  {
    id: 'tube',
    name: 'Dispenser Detection',
    category: 'Drug Discovery',
    description: 'Monitors laboratory equipment to detect if the dispenser has touched the tube correctly during automated processes',
    icon: 'tube',
    prompt: 'Monitor the laboratory dispenser and tube. Detect if the dispenser has touched or interacted with the tube during the automated process. Report the status of the dispenser-tube interaction.',
    backendValue: 'tube',
  },
];

const getIconComponent = (iconType: UseCase['icon']) => {
  const iconClass = "w-6 h-6";

  switch (iconType) {
    case 'bottle':
      return <Package className={iconClass} />;
    case 'motion':
      return <Eye className={iconClass} />;
    case 'security':
      return <Shield className={iconClass} />;
    case 'bed':
      return <BedDouble className={iconClass} />;
    case 'helmet':
      return <HardHat className={iconClass} />;
    case 'tube':
      return <TestTube2 className={iconClass} />;
    default:
      return <Eye className={iconClass} />;
  }
};

// Category color mapping
const categoryColors: Record<string, { bg: string; text: string; border: string }> = {
  'Security': { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
  'Production Line': { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  'Healthcare': { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  'Safety Compliance': { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
  'Drug Discovery': { bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
};

export default function UseCaseSelector({ selectedUseCase, onSelect, mode = 'demo' }: UseCaseSelectorProps) {
  // Filter use cases based on mode
  const availableUseCases = mode === 'live'
    ? useCases.filter(useCase => useCase.id === 'restricted-area')
    : useCases.filter(useCase => useCase.id !== 'restricted-area');

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-lg">
      <h3 className="text-xl font-semibold text-blue-800 mb-2">
        {mode === 'live' ? 'Live Stream Monitoring' : 'Video Analysis Use Cases'}
      </h3>
      <p className="text-sm text-gray-500 mb-5">
        {mode === 'live'
          ? 'Real-time security monitoring'
          : 'Select a use case to analyze your uploaded video'}
      </p>

      <div className={`grid gap-3 mb-5 ${mode === 'live' ? 'grid-cols-1' : 'grid-cols-2'}`}>
        {availableUseCases.map((useCase) => {
          const colors = categoryColors[useCase.category] || categoryColors['Security'];
          const isSelected = selectedUseCase?.id === useCase.id;

          return (
            <button
              key={useCase.id}
              onClick={() => onSelect(useCase)}
              className={`
                relative p-4 rounded-xl text-left transition-all duration-200 group
                ${isSelected
                  ? 'bg-blue-600 text-white border-2 border-blue-700 shadow-lg'
                  : 'bg-white hover:bg-gray-50 text-gray-800 border-2 border-gray-200 hover:border-blue-300 hover:shadow-md'
                }
              `}
            >
              <div className="flex items-start gap-3">
                <div className={`
                  p-3 rounded-lg flex-shrink-0
                  ${isSelected
                    ? 'bg-blue-700 text-white'
                    : 'bg-blue-100 text-blue-600 group-hover:bg-blue-200'
                  }
                `}>
                  {getIconComponent(useCase.icon)}
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-sm leading-tight mb-1">
                    {useCase.name}
                  </h4>
                  <span className={`
                    inline-block px-2 py-0.5 rounded-full text-xs font-medium
                    ${isSelected
                      ? 'bg-blue-500 text-blue-100'
                      : `${colors.bg} ${colors.text}`
                    }
                  `}>
                    {useCase.category}
                  </span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {selectedUseCase && (
        <div className="border-t border-gray-200 pt-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-gray-500">Selected:</span>
            <span className="text-sm font-semibold text-blue-700">{selectedUseCase.name}</span>
          </div>
          <p className="text-sm text-gray-600 leading-relaxed">
            {selectedUseCase.description}
          </p>
        </div>
      )}
    </div>
  );
}
