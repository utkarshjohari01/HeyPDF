import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { pingBackend } from '../api';

// ── Animated PDF + Chat bubble icon ──────────────────────────────────────────
function HeyPDFIcon() {
  return (
    <div className="relative flex items-center justify-center w-20 h-20 mx-auto mb-6">
      <div className="absolute inset-0 rounded-2xl bg-accent-500/30 blur-xl scale-110" />
      <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-accent-500 to-accent-700
                      flex items-center justify-center shadow-2xl shadow-accent-500/40">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 44" fill="none" className="w-10 h-11">
          <rect x="2" y="2" width="28" height="36" rx="3" fill="rgba(255,255,255,0.15)" stroke="white" strokeWidth="1.5"/>
          <line x1="8" y1="12" x2="24" y2="12" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
          <line x1="8" y1="17" x2="24" y2="17" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
          <line x1="8" y1="22" x2="18" y2="22" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
          <rect x="18" y="26" width="20" height="14" rx="7" fill="#4f8ef7" stroke="white" strokeWidth="1.2"/>
          <circle cx="24" cy="33" r="1.2" fill="white"/>
          <circle cx="28" cy="33" r="1.2" fill="white"/>
          <circle cx="32" cy="33" r="1.2" fill="white"/>
        </svg>
      </div>
    </div>
  );
}

// ── Animated gradient blobs ──────────────────────────────────────────────────
function GradientBlobs() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <div
        className="absolute -top-32 -left-32 w-[600px] h-[600px] rounded-full opacity-[0.12]"
        style={{
          background: 'radial-gradient(circle, #4f8ef7 0%, #7c3aed 60%, transparent 80%)',
          animation: 'blobDrift1 12s ease-in-out infinite alternate',
        }}
      />
      <div
        className="absolute -bottom-32 -right-32 w-[500px] h-[500px] rounded-full opacity-[0.10]"
        style={{
          background: 'radial-gradient(circle, #7c3aed 0%, #4f8ef7 60%, transparent 80%)',
          animation: 'blobDrift2 15s ease-in-out infinite alternate',
        }}
      />
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[450px] h-[450px] rounded-full opacity-[0.06]"
        style={{
          background: 'radial-gradient(circle, #4f8ef7 0%, transparent 70%)',
          animation: 'blobDrift3 18s ease-in-out infinite alternate',
        }}
      />
    </div>
  );
}

// ── Bouncing chevron at bottom ───────────────────────────────────────────────
function ScrollChevron() {
  return (
    <motion.div
      className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 opacity-40"
      animate={{ y: [0, 6, 0] }}
      transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
    >
      <span className="text-[0.65rem] text-gray-400 dark:text-gray-500 tracking-[0.15em] uppercase font-medium">
        Scroll down to learn more
      </span>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
        strokeWidth={2} stroke="currentColor" className="w-4 h-4 text-gray-400 dark:text-gray-500">
        <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
      </svg>
    </motion.div>
  );
}

// ── Disclaimer Modal (Step 2) ────────────────────────────────────────────────
function DisclaimerModal({ onComplete }) {
  const [serverStatus, setServerStatus] = useState('pinging'); // pinging, success, slow
  const [isStarting, setIsStarting] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const checkServer = async () => {
      const slowTimer = setTimeout(() => {
        if (isMounted && serverStatus === 'pinging') {
          setServerStatus('slow');
        }
      }, 10000);

      const awake = await pingBackend();
      clearTimeout(slowTimer);

      if (isMounted) {
        setServerStatus(awake ? 'success' : 'slow');
      }
    };
    checkServer();
    return () => { isMounted = false; };
  }, []);

  const handleLetsGo = async () => {
    setIsStarting(true);
    setTimeout(() => {
      onComplete();
    }, 1000);
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <motion.div
        className="w-full max-w-[480px] bg-white/70 dark:bg-white/5 backdrop-blur-xl
                   border border-white/12 rounded-3xl shadow-[0_24px_64px_rgba(0,0,0,0.4)] p-6 md:p-9
                   max-h-[90vh] overflow-y-auto"
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <div className="text-center mb-6">
          <p className="text-accent-500 text-xs font-bold uppercase tracking-widest mb-2">Before you begin</p>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">A Quick Note on Performance</h2>
          <div className="w-12 h-px bg-accent-500/50 mx-auto mt-4" />
        </div>

        <motion.div
          className="space-y-3 mb-8"
          variants={{ show: { transition: { staggerChildren: 0.1 } } }}
          initial="hidden"
          animate="show"
        >
          {/* Card 1 */}
          <motion.div variants={cardVariants} className="flex gap-4 p-4 rounded-xl border border-white/10 dark:border-white/5 bg-black/5 dark:bg-white/5 hover:border-black/10 dark:hover:border-white/20 transition-colors">
            <span className="text-2xl mt-0.5">📄</span>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-gray-100 text-sm mb-0.5">Size Matters</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">Please keep PDFs under 50KB due to Vercel Free Tier limitations.</p>
            </div>
          </motion.div>

          {/* Card 2 */}
          <motion.div variants={cardVariants} className="flex gap-4 p-4 rounded-xl border border-white/10 dark:border-white/5 bg-black/5 dark:bg-white/5 hover:border-black/10 dark:hover:border-white/20 transition-colors">
            <span className="text-2xl mt-0.5">🌏</span>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-gray-100 text-sm mb-0.5">Global Journey</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">Our servers are in Singapore! It might take a moment to wake up and connect, so grab a coffee ☕</p>
            </div>
          </motion.div>

          {/* Card 3 */}
          <motion.div variants={cardVariants} className="flex gap-4 p-4 rounded-xl border border-white/10 dark:border-white/5 bg-black/5 dark:bg-white/5 hover:border-black/10 dark:hover:border-white/20 transition-colors">
            <span className="text-2xl mt-0.5">🚀</span>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-gray-100 text-sm mb-0.5">Full Power</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                To see the true potential of HeyPDF without limits, run it locally! Check out the{' '}
                <a href="https://github.com/utkarshjohari01/HeyPDF" target="_blank" rel="noreferrer" className="text-accent-500 hover:underline">GitHub</a>.
              </p>
            </div>
          </motion.div>
        </motion.div>

        <div className="flex flex-col gap-4">
          <motion.button
            onClick={handleLetsGo}
            disabled={isStarting}
            className="relative w-full py-4 rounded-full bg-accent-500 text-white font-bold text-base
                       shadow-lg shadow-accent-500/30 overflow-hidden flex items-center justify-center
                       focus:outline-none disabled:opacity-90 disabled:cursor-not-allowed"
            whileHover={!isStarting ? { scale: 1.03, boxShadow: '0 0 24px rgba(79,142,247,0.6)' } : {}}
            whileTap={!isStarting ? { scale: 0.97 } : {}}
          >
            {isStarting ? (
               <svg className="w-6 h-6 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
               </svg>
            ) : (
              "LET'S GO! 🚀"
            )}
          </motion.button>

          {/* Server Status Indicator */}
          <div className="text-center h-4">
            {serverStatus === 'pinging' && <p className="text-xs font-medium text-amber-500 dark:text-amber-400 animate-pulse">🟡 Waking up server...</p>}
            {serverStatus === 'success' && <p className="text-xs font-medium text-emerald-500 dark:text-emerald-400">🟢 Server is ready!</p>}
            {serverStatus === 'slow' && <p className="text-xs font-medium text-rose-500 dark:text-rose-400">🔴 Server is slow today, please be patient</p>}
          </div>
        </div>

      </motion.div>
    </div>
  );
}

// ── Main Landing Page ────────────────────────────────────────────────────────
export default function LandingPage({ onComplete }) {
  const [showModal, setShowModal] = useState(false);

  // Stagger variants
  const container = {
    hidden: {},
    show: { transition: { staggerChildren: 0.15 } },
  };
  const item = {
    hidden: { opacity: 0, y: 24 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
  };

  return (
    <motion.div
      className="relative min-h-screen flex flex-col items-center justify-center
                 bg-surface-50 dark:bg-navy-900 overflow-hidden px-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.5 } }}
    >
      {/* Animated background blobs */}
      <GradientBlobs />

      {/* Center content */}
      <motion.div
        className="relative z-10 flex flex-col items-center text-center max-w-lg w-full"
        variants={container}
        initial="hidden"
        animate="show"
      >
        <motion.div variants={item}>
          <HeyPDFIcon />
        </motion.div>

        <motion.h1
          variants={item}
          className="text-[2.8rem] md:text-[4rem] font-extrabold leading-[1.1] tracking-tight mb-3"
          style={{
            background: 'linear-gradient(90deg, #4f8ef7 0%, #93bbfb 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          HeyPDF
        </motion.h1>

        <motion.p
          variants={item}
          className="text-lg md:text-xl font-medium text-gray-500 dark:text-gray-400
                     tracking-[0.04em] mb-2"
        >
          Chat with your PDFs using AI
        </motion.p>

        <motion.div variants={item} className="flex flex-col items-center mb-10">
          <div className="w-16 h-px bg-gray-300 dark:bg-navy-600 mb-3 mt-1" />
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Built by{' '}
            <a
              href="https://github.com/utkarshjohari01"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-500 hover:underline font-medium"
            >
              Utkarsh Johari
            </a>
          </p>
        </motion.div>

        <motion.button
          variants={item}
          onClick={() => setShowModal(true)}
          className="inline-flex items-center gap-3 font-bold text-white
                     bg-accent-500 rounded-full px-12 py-4 text-base
                     shadow-lg shadow-accent-500/30 cursor-pointer
                     focus:outline-none focus:ring-2 focus:ring-accent-400 focus:ring-offset-2"
          whileHover={{
            scale: 1.05,
            boxShadow: '0 0 24px rgba(79,142,247,0.66)',
          }}
          whileTap={{ scale: 0.97 }}
          transition={{ type: 'spring', stiffness: 400, damping: 20 }}
        >
          GET STARTED
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"
            fill="currentColor" className="w-5 h-5">
            <path fillRule="evenodd"
              d="M3 10a.75.75 0 0 1 .75-.75h10.638L10.23 5.29a.75.75 0 1 1 1.04-1.08l5.5 5.25a.75.75 0 0 1 0 1.08l-5.5 5.25a.75.75 0 1 1-1.04-1.08l4.158-3.96H3.75A.75.75 0 0 1 3 10Z"
              clipRule="evenodd" />
          </svg>
        </motion.button>
      </motion.div>

      <ScrollChevron />

      {/* Modal Overlay */}
      <AnimatePresence>
        {showModal && <DisclaimerModal onComplete={onComplete} />}
      </AnimatePresence>

      <style>{`
        @keyframes blobDrift1 {
          0%   { transform: translate(0px, 0px) scale(1); }
          100% { transform: translate(60px, 40px) scale(1.1); }
        }
        @keyframes blobDrift2 {
          0%   { transform: translate(0px, 0px) scale(1); }
          100% { transform: translate(-50px, -30px) scale(1.08); }
        }
        @keyframes blobDrift3 {
          0%   { transform: translate(-50%, -50%) scale(1); }
          100% { transform: translate(calc(-50% + 30px), calc(-50% + 20px)) scale(1.05); }
        }
      `}</style>
    </motion.div>
  );
}
