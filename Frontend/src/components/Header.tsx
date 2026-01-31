// Import logo
import logo from '../assets/logo.png';

export default function Header() {
  return (
    <div className="pt-6 pb-4 px-6">
      <div className="flex items-center gap-4">
        <img
          src={logo}
          alt="OcchioAI Logo"
          className="h-14 w-auto object-contain drop-shadow-md"
        />
        <div className="flex flex-col">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-700 to-cyan-500 bg-clip-text text-transparent">
            OcchioAI
          </h1>
          <span className="text-xs text-gray-500 font-medium tracking-wide">
            Intelligent Video Analytics
          </span>
        </div>
      </div>
    </div>
  );
}
