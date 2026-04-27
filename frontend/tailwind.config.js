/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      colors: {
        // Dark theme palette
        navy: {
          950: '#070b12',
          900: '#0f1117',
          800: '#161b27',
          700: '#1e2639',
          600: '#263048',
        },
        // Accent electric blue
        accent: {
          300: '#93bbfb',
          400: '#6a9ff9',
          500: '#4f8ef7',
          600: '#3a7af3',
          700: '#2563eb',
        },
        // Light theme
        surface: {
          50:  '#f5f7fa',
          100: '#eef0f5',
          200: '#e1e5ee',
          300: '#c8cfe0',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.35s ease-out forwards',
        'fade-in': 'fadeIn 0.2s ease-out forwards',
        'spin-slow': 'spin 1.5s linear infinite',
        'pulse-soft': 'pulseSoft 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
