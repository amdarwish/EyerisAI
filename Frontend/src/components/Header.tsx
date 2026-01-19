import { Activity } from 'lucide-react';
// Import logo
import logo from '../assets/logo.png';

export default function Header() {
  return (
    <div className="pt-8 pb-6 px-6">
      <div className="flex items-center gap-4">
        <div className="flex items-center justify-center w-14 h-14 bg-white rounded-2xl shadow-lg border border-gray-100">
          <img 
            src={logo} 
            alt="AbuVision Logo" 
            className="w-10 h-10 object-contain"
          />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-gray-800">AbuVision</h1>
        </div>
      </div>
    </div>
  );
}
