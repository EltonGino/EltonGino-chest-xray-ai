import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

// ── Design tokens ────────────────────────────────────────────────────────────
const css = `
  :root {
    --bg-void:      #030711;
    --bg-surface:   #060F1E;
    --bg-elevated:  #0A1628;
    --bg-card:      #0D1E35;
    --border:       #12274A;
    --border-lit:   #1E3A5F;
    --accent:       #00C2FF;
    --accent-dim:   rgba(0,194,255,0.12);
    --accent-glow:  rgba(0,194,255,0.35);
    --indigo:       #6366F1;
    --success:      #10B981;
    --warning:      #F59E0B;
    --danger:       #EF4444;
    --text-1:       #E2EFF8;
    --text-2:       #7B9DB8;
    --text-3:       #3D6482;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'DM Sans', sans-serif;
  }

  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 var(--accent-glow); }
    70%  { box-shadow: 0 0 0 12px transparent; }
    100% { box-shadow: 0 0 0 0 transparent; }
  }

  @keyframes scan {
    0%   { top: 0%; opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
  }

  @keyframes grid-drift {
    0%   { transform: translateY(0); }
    100% { transform: translateY(40px); }
  }

  @keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
  }

  .scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    box-shadow: 0 0 20px var(--accent-glow);
    animation: scan 1.8s ease-in-out;
    pointer-events: none;
    z-index: 10;
  }

  .grid-bg {
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.35;
    animation: grid-drift 12s linear infinite alternate;
    pointer-events: none;
    z-index: 0;
  }

  .shimmer-bar {
    background: linear-gradient(90deg,
      var(--bg-elevated) 0%,
      var(--border-lit) 50%,
      var(--bg-elevated) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
  }

  .report-text {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    line-height: 1.9;
    color: var(--text-2);
    white-space: pre-wrap;
  }

  .report-text strong, .report-text b {
    color: var(--accent);
    font-weight: 500;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .badge-accent {
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid rgba(0,194,255,0.25);
  }

  .badge-success {
    background: rgba(16,185,129,0.1);
    color: var(--success);
    border: 1px solid rgba(16,185,129,0.25);
  }
`

// ── Severity helpers ─────────────────────────────────────────────────────────
function getSeverity(prob) {
  if (prob >= 0.6) return { color: '#EF4444', label: 'HIGH',     glow: 'rgba(239,68,68,0.4)' }
  if (prob >= 0.35) return { color: '#F59E0B', label: 'MODERATE', glow: 'rgba(245,158,11,0.4)' }
  if (prob >= 0.15) return { color: '#00C2FF', label: 'LOW',      glow: 'rgba(0,194,255,0.4)' }
  return              { color: '#3D6482',   label: 'TRACE',    glow: 'transparent' }
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Header({ modelInfo }) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      style={{
        position: 'relative', zIndex: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '20px 40px',
        borderBottom: '1px solid var(--border)',
        background: 'linear-gradient(180deg, rgba(6,15,30,0.95) 0%, transparent 100%)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        {/* Logo mark */}
        <div style={{ position: 'relative' }}>
          <motion.div
            animate={{ scale: [1, 1.15, 1] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              width: 36, height: 36,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(0,194,255,0.5) 0%, rgba(0,194,255,0.05) 70%)',
              border: '1.5px solid var(--accent)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </motion.div>
        </div>
        <div>
          <div style={{
            fontFamily: 'var(--font-display)', fontSize: '1.1rem',
            fontWeight: 700, letterSpacing: '0.08em',
            color: 'var(--text-1)',
          }}>
            CHEST<span style={{ color: 'var(--accent)' }}>·</span>AI
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.62rem',
            color: 'var(--text-3)', letterSpacing: '0.12em',
          }}>
            RADIOLOGICAL ANALYSIS SYSTEM
          </div>
        </div>
      </div>

      {modelInfo && (
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="badge badge-accent">{modelInfo.architecture}</span>
          <span className="badge badge-success">AUC {modelInfo.best_auc?.toFixed(3)}</span>
        </div>
      )}
    </motion.header>
  )
}


function UploadZone({ onUpload, loading }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onUpload(file)
  }, [onUpload])

  const handleFile = (e) => {
    const file = e.target.files[0]
    if (file) onUpload(file)
  }

  return (
    <motion.div
      onClick={() => !loading && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      animate={{
        borderColor: dragging ? 'var(--accent)' : 'var(--border-lit)',
        boxShadow: dragging
          ? '0 0 40px var(--accent-glow), inset 0 0 40px var(--accent-dim)'
          : '0 0 0 transparent',
      }}
      transition={{ duration: 0.2 }}
      style={{
        width: '100%', maxWidth: 420,
        aspectRatio: '1 / 1.1',
        border: '1.5px dashed var(--border-lit)',
        borderRadius: 16,
        background: 'var(--bg-surface)',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: 20, cursor: loading ? 'not-allowed' : 'pointer',
        position: 'relative', overflow: 'hidden',
        userSelect: 'none',
      }}
    >
      {/* Background grid pattern inside zone */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'radial-gradient(circle at center, rgba(0,194,255,0.03) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Corner accents */}
      {['0,0', '0,auto', 'auto,0', 'auto,auto'].map((pos, i) => {
        const [top, bottom] = pos.split(',')
        return (
          <div key={i} style={{
            position: 'absolute',
            top: top === '0' ? 12 : 'auto',
            bottom: top !== '0' ? 12 : 'auto',
            left: i % 2 === 0 ? 12 : 'auto',
            right: i % 2 !== 0 ? 12 : 'auto',
            width: 16, height: 16,
            borderTop: (i < 2) ? '1.5px solid var(--accent)' : 'none',
            borderBottom: (i >= 2) ? '1.5px solid var(--accent)' : 'none',
            borderLeft: (i % 2 === 0) ? '1.5px solid var(--accent)' : 'none',
            borderRight: (i % 2 !== 0) ? '1.5px solid var(--accent)' : 'none',
          }} />
        )
      })}

      {/* Icon */}
      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          width: 64, height: 64,
          borderRadius: '50%',
          background: 'var(--accent-dim)',
          border: '1px solid rgba(0,194,255,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeLinecap="round" strokeLinejoin="round"/>
          <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round"/>
          <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round"/>
        </svg>
      </motion.div>

      <div style={{ textAlign: 'center', padding: '0 24px' }}>
        <div style={{
          fontFamily: 'var(--font-display)', fontSize: '0.95rem',
          fontWeight: 600, color: 'var(--text-1)', marginBottom: 6,
        }}>
          Upload Chest X-Ray
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.68rem',
          color: 'var(--text-3)', lineHeight: 1.8,
          letterSpacing: '0.04em',
        }}>
          DRAG & DROP OR CLICK TO SELECT<br/>
          PNG · JPG · WEBP · DICOM-EXPORT
        </div>
      </div>

      <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFile} />
    </motion.div>
  )
}


function ImagePreview({ src, scanning }) {
  return (
    <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden' }}>
      <img
        src={src}
        alt="X-Ray Preview"
        style={{ width: '100%', maxWidth: 420, display: 'block', borderRadius: 12 }}
      />
      <AnimatePresence>
        {scanning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ position: 'absolute', inset: 0 }}
          >
            <div className="scan-line" />
            <div style={{
              position: 'absolute', inset: 0,
              background: 'rgba(0,194,255,0.04)',
            }} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}


function PredictionBars({ predictions }) {
  const entries = Object.entries(predictions)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {entries.map(([cls, prob], i) => {
        const sev = getSeverity(prob)
        const pct = Math.round(prob * 100)

        return (
          <motion.div
            key={cls}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04, duration: 0.35, ease: 'easeOut' }}
            style={{
              display: 'grid',
              gridTemplateColumns: '130px 1fr 48px',
              alignItems: 'center',
              gap: 10,
              padding: '5px 0',
            }}
          >
            {/* Class name */}
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
              color: prob > 0.15 ? 'var(--text-1)' : 'var(--text-3)',
              fontWeight: prob > 0.35 ? 500 : 400,
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {cls}
            </div>

            {/* Bar track */}
            <div style={{
              height: 5, borderRadius: 3,
              background: 'var(--bg-elevated)',
              overflow: 'hidden', position: 'relative',
            }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ delay: i * 0.04 + 0.1, duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                style={{
                  height: '100%', borderRadius: 3,
                  background: prob > 0.35
                    ? `linear-gradient(90deg, ${sev.color}88, ${sev.color})`
                    : `linear-gradient(90deg, ${sev.color}44, ${sev.color}66)`,
                  boxShadow: prob > 0.35 ? `0 0 8px ${sev.glow}` : 'none',
                }}
              />
            </div>

            {/* Percentage */}
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '0.68rem',
              color: sev.color, textAlign: 'right',
              fontWeight: 500,
            }}>
              {pct}%
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}


function GradCamViewer({ originalSrc, gradcamSrc, gradcamClass }) {
  const [showCam, setShowCam] = useState(false)

  return (
    <div>
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', marginBottom: 12,
      }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.68rem',
          color: 'var(--text-3)', letterSpacing: '0.08em',
        }}>
          {showCam
            ? `GRAD-CAM → ${gradcamClass?.toUpperCase()}`
            : 'ORIGINAL IMAGE'
          }
        </div>
        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          onClick={() => setShowCam(v => !v)}
          style={{
            padding: '5px 14px',
            background: showCam ? 'var(--accent-dim)' : 'transparent',
            border: '1px solid var(--border-lit)',
            borderRadius: 20,
            color: showCam ? 'var(--accent)' : 'var(--text-2)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.65rem', letterSpacing: '0.06em',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {showCam ? '← ORIGINAL' : 'GRAD-CAM →'}
        </motion.button>
      </div>

      <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden' }}>
        <AnimatePresence mode="wait">
          <motion.img
            key={showCam ? 'cam' : 'orig'}
            src={showCam ? `data:image/png;base64,${gradcamSrc}` : originalSrc}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            style={{ width: '100%', display: 'block', borderRadius: 10 }}
          />
        </AnimatePresence>

        {showCam && (
          <div style={{
            position: 'absolute', bottom: 10, right: 10,
            fontFamily: 'var(--font-mono)', fontSize: '0.6rem',
            color: '#fff', background: 'rgba(0,0,0,0.6)',
            padding: '3px 8px', borderRadius: 4,
            backdropFilter: 'blur(4px)',
          }}>
            INFERNO COLORMAP
          </div>
        )}
      </div>
    </div>
  )
}


function ReportPanel({ report }) {
  const [displayed, setDisplayed] = useState('')
  const [done, setDone] = useState(false)
  const indexRef = useRef(0)

  // Typewriter effect
  useState(() => {
    if (!report) return
    indexRef.current = 0
    setDisplayed('')
    setDone(false)

    const interval = setInterval(() => {
      indexRef.current++
      setDisplayed(report.slice(0, indexRef.current))
      if (indexRef.current >= report.length) {
        clearInterval(interval)
        setDone(true)
      }
    }, 12)

    return () => clearInterval(interval)
  })

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
        borderRadius: 12,
        padding: 20,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', marginBottom: 16,
      }}>
        <div style={{
          fontFamily: 'var(--font-display)', fontSize: '0.78rem',
          fontWeight: 600, letterSpacing: '0.1em',
          color: 'var(--text-1)',
        }}>
          RADIOLOGY REPORT
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <motion.div
            animate={done ? { opacity: 0 } : { opacity: [1, 0, 1] }}
            transition={{ duration: 0.8, repeat: done ? 0 : Infinity }}
            style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--accent)',
            }}
          />
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.62rem',
            color: done ? 'var(--success)' : 'var(--accent)',
            letterSpacing: '0.08em',
          }}>
            {done ? 'COMPLETE' : 'GENERATING'}
          </span>
        </div>
      </div>

      {/* Divider */}
      <div style={{
        height: 1, background: 'var(--border)', marginBottom: 16,
      }} />

      {/* Report text */}
      <div
        className="report-text"
        dangerouslySetInnerHTML={{
          __html: displayed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        }}
      />

      {/* Cursor */}
      {!done && (
        <motion.span
          animate={{ opacity: [1, 0] }}
          transition={{ duration: 0.5, repeat: Infinity }}
          style={{
            display: 'inline-block', width: 2, height: '0.85em',
            background: 'var(--accent)', marginLeft: 2,
            verticalAlign: 'middle',
          }}
        />
      )}

      {/* Bottom glow */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        height: 1,
        background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
        opacity: 0.3,
      }} />
    </motion.div>
  )
}


function LoadingState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', gap: 20, padding: 40,
      }}
    >
      {/* Spinner */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        style={{
          width: 48, height: 48,
          border: '2px solid var(--border)',
          borderTop: '2px solid var(--accent)',
          borderRadius: '50%',
        }}
      />
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: '0.72rem',
        color: 'var(--text-2)', letterSpacing: '0.1em',
      }}>
        ANALYSING RADIOGRAPH
      </div>

      {/* Skeleton bars */}
      <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {[80, 60, 45, 70, 35].map((w, i) => (
          <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <div className="shimmer-bar" style={{ width: 110, height: 8, borderRadius: 4 }} />
            <div className="shimmer-bar" style={{ flex: 1, height: 5, borderRadius: 3 }} />
            <div className="shimmer-bar" style={{ width: 32, height: 8, borderRadius: 4 }} />
          </div>
        ))}
      </div>
    </motion.div>
  )
}


// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [modelInfo, setModelInfo]     = useState(null)
  const [imageFile, setImageFile]     = useState(null)
  const [imageUrl, setImageUrl]       = useState(null)
  const [scanning, setScanning]       = useState(false)
  const [loading, setLoading]         = useState(false)
  const [result, setResult]           = useState(null)
  const [error, setError]             = useState(null)

  // Fetch model info once
  useState(() => {
    axios.get('/health')
      .then(r => setModelInfo(r.data))
      .catch(() => {})
  })

  const handleUpload = useCallback(async (file) => {
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setScanning(true)
    setLoading(true)

    // Scan animation duration
    setTimeout(() => setScanning(false), 1800)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('generate_report', 'true')

    try {
      const { data } = await axios.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
      if (data.meta) setModelInfo(prev => ({ ...prev, ...data.meta }))
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Is the API running?')
    } finally {
      setLoading(false)
    }
  }, [])

  const hasResults = result && !loading

  return (
    <>
      <style>{css}</style>

      {/* Background grid */}
      <div className="grid-bg" />

      {/* Radial glow */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 60% 40% at 50% 0%, rgba(0,194,255,0.06) 0%, transparent 70%)',
      }} />

      {/* App shell */}
      <div style={{ position: 'relative', zIndex: 1, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

        <Header modelInfo={modelInfo} />

        {/* Main content */}
        <main style={{
          flex: 1,
          display: 'flex',
          alignItems: hasResults ? 'flex-start' : 'center',
          justifyContent: 'center',
          padding: '40px 24px',
          gap: 32,
          transition: 'align-items 0.4s',
        }}>

          {/* Left column: upload / image */}
          <motion.div
            layout
            style={{
              display: 'flex', flexDirection: 'column', gap: 20,
              width: '100%', maxWidth: 420,
              flexShrink: 0,
            }}
          >
            {/* Section label */}
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
              color: 'var(--text-3)', letterSpacing: '0.12em',
            }}>
              01 / INPUT
            </div>

            <AnimatePresence mode="wait">
              {!imageUrl ? (
                <motion.div key="upload" initial={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <UploadZone onUpload={handleUpload} loading={loading} />
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
                >
                  {hasResults && result.gradcam ? (
                    <GradCamViewer
                      originalSrc={imageUrl}
                      gradcamSrc={result.gradcam}
                      gradcamClass={result.gradcam_class}
                    />
                  ) : (
                    <ImagePreview src={imageUrl} scanning={scanning} />
                  )}

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      setImageUrl(null)
                      setResult(null)
                      setError(null)
                    }}
                    style={{
                      padding: '10px',
                      background: 'transparent',
                      border: '1px solid var(--border-lit)',
                      borderRadius: 8,
                      color: 'var(--text-2)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.68rem', letterSpacing: '0.06em',
                      cursor: 'pointer',
                    }}
                  >
                    ← UPLOAD NEW IMAGE
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  style={{
                    padding: '12px 16px',
                    background: 'rgba(239,68,68,0.08)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: 8,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.72rem',
                    color: 'var(--danger)',
                  }}
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Right column: results */}
          <AnimatePresence>
            {(loading || hasResults) && (
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 30 }}
                transition={{ duration: 0.45, ease: 'easeOut' }}
                style={{
                  flex: 1, maxWidth: 580,
                  display: 'flex', flexDirection: 'column', gap: 20,
                }}
              >
                {/* Section label */}
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
                  color: 'var(--text-3)', letterSpacing: '0.12em',
                }}>
                  02 / ANALYSIS
                </div>

                {loading ? (
                  <div style={{
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border)',
                    borderRadius: 12, padding: 20,
                  }}>
                    <LoadingState />
                  </div>
                ) : (
                  <>
                    {/* Predictions card */}
                    <motion.div
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      style={{
                        background: 'var(--bg-surface)',
                        border: '1px solid var(--border)',
                        borderRadius: 12, padding: 20,
                      }}
                    >
                      <div style={{
                        display: 'flex', alignItems: 'center',
                        justifyContent: 'space-between', marginBottom: 16,
                      }}>
                        <div style={{
                          fontFamily: 'var(--font-display)', fontSize: '0.78rem',
                          fontWeight: 600, letterSpacing: '0.1em', color: 'var(--text-1)',
                        }}>
                          FINDING PROBABILITIES
                        </div>
                        <div style={{ display: 'flex', gap: 6 }}>
                          {[
                            { color: 'var(--danger)',  label: 'HIGH' },
                            { color: 'var(--warning)', label: 'MOD' },
                            { color: 'var(--accent)',  label: 'LOW' },
                          ].map(({ color, label }) => (
                            <div key={label} style={{
                              display: 'flex', alignItems: 'center', gap: 4,
                              fontFamily: 'var(--font-mono)', fontSize: '0.58rem',
                              color: 'var(--text-3)',
                            }}>
                              <div style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
                              {label}
                            </div>
                          ))}
                        </div>
                      </div>

                      <div style={{ height: 1, background: 'var(--border)', marginBottom: 14 }} />

                      <PredictionBars predictions={result.predictions} />
                    </motion.div>

                    {/* Report card */}
                    {result.report && <ReportPanel report={result.report} />}
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty state hint */}
          {!imageUrl && !loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              style={{
                position: 'absolute', bottom: 32,
                fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
                color: 'var(--text-3)', letterSpacing: '0.08em',
                textAlign: 'center',
              }}
            >
              EfficientNetV2-S · 15-class multi-label · NIH ChestX-ray14
            </motion.div>
          )}
        </main>
      </div>
    </>
  )
}
