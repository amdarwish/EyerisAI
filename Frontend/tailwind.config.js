/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e6f1ff',
          100: '#b3d7ff',
          200: '#80bdff',
          300: '#4da3ff',
          400: '#1a89ff',
          500: '#0066cc',
          600: '#0052a3',
          700: '#003d7a',
          800: '#002952',
          900: '#001429',
        },
        accent: {
          cyan: '#00d4ff',
          blue: '#0099ff',
          dark: '#001a33',
        },
        status: {
          idle: '#64748b',
          processing: '#0099ff',
          alert: '#ef4444',
          success: '#10b981',
        }
      },
    },
  },
  plugins: [],
};
