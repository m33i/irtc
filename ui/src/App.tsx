import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Viewer, Entity, CameraFlyTo, type CesiumComponentRef } from 'resium'
import * as Cesium from 'cesium'
import {
  Ion, Cartesian2, Cartesian3, Color,
  UrlTemplateImageryProvider, ImageryLayer,
  SingleTileImageryProvider, Rectangle, VerticalOrigin,
} from 'cesium'
import { analyzeImage } from './api'
import type { Analysis, Match, PipelineStatus } from './types'

Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_TOKEN ?? ''

// ── Imagery layers ────────────────────────────────────────────────────────────

const BASE_LAYER = new ImageryLayer(
  new UrlTemplateImageryProvider({
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    credit: 'Tiles © Esri',
    maximumLevel: 19,
  })
)

const LABELS_LAYER = new ImageryLayer(
  new UrlTemplateImageryProvider({
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    credit: '',
    maximumLevel: 19,
  })
)

// ── MODIS cloud slider ────────────────────────────────────────────────────────

const CLOUD_EPOCH_MS = new Date('2015-01-01').getTime()
const CLOUD_MAX_DAYS = Math.max(0, Math.floor((Date.now() - 86400000 * 2 - CLOUD_EPOCH_MS) / 86400000))

const dayToDate = (day: number) =>
  new Date(CLOUD_EPOCH_MS + day * 86400000).toISOString().slice(0, 10)

// Terra: ~10:30 local overpass → good for morning hours
// Aqua:  ~13:30 local overpass → good for afternoon hours
const gibsTerra = (date: string) =>
  `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`
const gibsAqua = (date: string) =>
  `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Aqua_CorrectedReflectance_TrueColor/default/${date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`
const gibsUrl = (date: string, hour: number) => hour < 12 ? gibsTerra(date) : gibsAqua(date)

// ── Match pins — 📌 emoji rendered on canvas ──────────────────────────────────

const PUSHPIN = (() => {
  const size = 32, pad = 4, total = size + pad * 2
  const canvas = document.createElement('canvas')
  canvas.width = total; canvas.height = total
  const ctx = canvas.getContext('2d')!
  ctx.font = `${size}px serif`
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText('📌', total / 2, total / 2)
  return canvas.toDataURL()
})()

// ── Primitives ────────────────────────────────────────────────────────────────

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <p className="text-[15px] font-medium text-white mb-3">{children}</p>
}

function Card({ children, className = '', noPad = false }: {
  children: React.ReactNode; className?: string; noPad?: boolean
}) {
  return (
    <div
      className={`w-full rounded-xl bg-[#252528] overflow-hidden ${noPad ? '' : 'p-4'} ${className}`}
      style={{ boxShadow: '0 4px 20px rgba(0,0,0,0.55)' }}
    >
      {children}
    </div>
  )
}

function KVRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-baseline py-1.5 border-b border-white/5 last:border-0">
      <span className="text-[13px] text-white/40">{label}</span>
      <span className="text-[13px] text-white tabular-nums">{value}</span>
    </div>
  )
}

function Btn({ children, onClick, active = false, className = '' }: {
  children: React.ReactNode; onClick?: () => void; active?: boolean; className?: string
}) {
  return (
    <button
      onClick={onClick}
      className={`text-[12px] font-medium px-3.5 py-1.5 rounded-lg transition-all
        ${active ? 'bg-white/20 text-white' : 'bg-white/8 text-white/55 hover:bg-white/14 hover:text-white/90'}
        ${className}`}
    >
      {children}
    </button>
  )
}

// ── Cloud slider ──────────────────────────────────────────────────────────────

// ── Custom mini-calendar ──────────────────────────────────────────────────────

const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function MiniCalendar({ date, onChange }: {
  date: string   // YYYY-MM-DD
  onChange: (d: string) => void
}) {
  const [year, month, selDay] = date.split('-').map(Number)
  const [viewing, setViewing] = useState({ year, month })
  void selDay // used implicitly via date string comparison

  const firstDay = new Date(viewing.year, viewing.month - 1, 1).getDay()
  const daysInMonth = new Date(viewing.year, viewing.month, 0).getDate()
  const today = new Date().toISOString().slice(0, 10)

  const prevMonth = () => setViewing(v => ({
    year: v.month === 1 ? v.year - 1 : v.year,
    month: v.month === 1 ? 12 : v.month - 1,
  }))
  const nextMonth = () => setViewing(v => ({
    year: v.month === 12 ? v.year + 1 : v.year,
    month: v.month === 12 ? 1 : v.month + 1,
  }))
  const prevYear = () => setViewing(v => ({ ...v, year: Math.max(2015, v.year - 1) }))
  const nextYear = () => setViewing(v => ({ ...v, year: Math.min(new Date().getFullYear(), v.year + 1) }))

  const select = (d: number) => {
    const s = `${viewing.year}-${String(viewing.month).padStart(2,'0')}-${String(d).padStart(2,'0')}`
    if (s <= today && s >= '2015-01-01') onChange(s)
  }

  const cells: (number | null)[] = [...Array(firstDay).fill(null), ...Array.from({length: daysInMonth}, (_,i) => i+1)]

  return (
    <div className="rounded-xl p-3 select-none"
      style={{ background: 'rgba(255,255,255,0.06)', minWidth: 200 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2 gap-1">
        <button onClick={prevYear}  className="text-white/30 hover:text-white text-[13px] leading-none transition-colors px-0.5">«</button>
        <button onClick={prevMonth} className="text-white/45 hover:text-white text-[13px] leading-none transition-colors px-0.5">‹</button>
        <span className="text-[12px] font-medium text-white/80 flex-1 text-center">
          {MONTH_NAMES[viewing.month-1]} {viewing.year}
        </span>
        <button onClick={nextMonth} className="text-white/45 hover:text-white text-[13px] leading-none transition-colors px-0.5">›</button>
        <button onClick={nextYear}  className="text-white/30 hover:text-white text-[13px] leading-none transition-colors px-0.5">»</button>
      </div>
      {/* Day headers */}
      <div className="grid grid-cols-7 mb-1">
        {['Su','Mo','Tu','We','Th','Fr','Sa'].map(d => (
          <span key={d} className="text-center text-[10px] text-white/25">{d}</span>
        ))}
      </div>
      {/* Day grid */}
      <div className="grid grid-cols-7 gap-y-0.5">
        {cells.map((d, i) => {
          if (!d) return <span key={`pad-${i}`} />
          const dateStr = `${viewing.year}-${String(viewing.month).padStart(2,'0')}-${String(d).padStart(2,'0')}`
          const isSelected = dateStr === date
          const isFuture = dateStr > today || dateStr < '2015-01-01'
          return (
            <button
              key={dateStr}
              onClick={() => select(d)}
              disabled={isFuture}
              className={`text-[11px] rounded-md py-0.5 transition-colors leading-tight
                ${isSelected ? 'bg-white/25 text-white font-medium' : ''}
                ${isFuture ? 'text-white/15 cursor-default' : isSelected ? '' : 'text-white/55 hover:bg-white/12 hover:text-white'}`}
            >
              {d}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Cloud controls overlay ────────────────────────────────────────────────────

function CloudControls({ day, onDay, hour, onHour, visible, onToggle }: {
  day: number; onDay: (d: number) => void
  hour: number; onHour: (h: number) => void
  visible: boolean; onToggle: () => void
}) {
  const [calOpen, setCalOpen] = useState(false)
  const date = dayToDate(day)

  return (
    <div className="relative flex items-center gap-3 rounded-2xl px-4 py-2.5 select-none"
      style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.08)' }}>

      {/* Date button → opens custom calendar */}
      <button
        onClick={() => setCalOpen(o => !o)}
        className={`text-[12px] font-mono px-3 py-1.5 rounded-lg border transition-all
          ${calOpen ? 'border-white/30 bg-white/12 text-white' : 'border-white/10 bg-white/6 text-white/70 hover:border-white/20 hover:text-white/90'}`}
      >
        {date}
      </button>

      {/* Custom calendar popover */}
      {calOpen && (
        <div className="absolute bottom-full mb-2 left-0 z-20"
          style={{ filter: 'drop-shadow(0 8px 24px rgba(0,0,0,0.7))' }}>
          <div className="rounded-2xl overflow-hidden border border-white/10"
            style={{ background: 'rgba(18,18,22,0.97)', backdropFilter: 'blur(10px)' }}>
            <MiniCalendar
              date={date}
              onChange={d => {
                const ms = new Date(d).getTime()
                onDay(Math.round((ms - CLOUD_EPOCH_MS) / 86400000))
                setCalOpen(false)
              }}
            />
          </div>
        </div>
      )}

      {/* Hour slider */}
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-white/35 font-mono w-12 text-right tabular-nums">
          {String(hour).padStart(2,'0')}:00
        </span>
        <input
          type="range" min={0} max={23} value={hour} step={1}
          onChange={e => onHour(parseInt(e.target.value))}
          className="w-36 cursor-pointer"
          style={{ accentColor: 'rgba(255,255,255,0.55)' }}
        />
        <span className="text-[11px] font-mono"
          style={{ color: hour < 12 ? 'rgba(251,191,36,0.7)' : 'rgba(147,197,253,0.7)', minWidth: 32 }}>
          {hour < 12 ? 'Terra' : 'Aqua'}
        </span>
      </div>

      <Btn onClick={onToggle} active={visible}>
        {visible ? 'Clouds ●' : 'Clouds ○'}
      </Btn>
    </div>
  )
}

// ── Photo modal ───────────────────────────────────────────────────────────────

function PhotoModal({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.88)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <img
        src={src} alt="cloud"
        className="max-w-[90vw] max-h-[90vh] object-contain rounded-xl"
        style={{ boxShadow: '0 8px 40px rgba(0,0,0,0.8)' }}
        onClick={e => e.stopPropagation()}
      />
      <button
        onClick={onClose}
        className="absolute top-5 right-6 text-white/50 hover:text-white text-2xl leading-none transition-colors"
      >
        ×
      </button>
    </div>
  )
}

// ── Photo area ────────────────────────────────────────────────────────────────

function PhotoArea({ preview, isRunning, step, total, message, isError, errorMsg, onFile, onExpand }: {
  preview: string | null
  isRunning: boolean; step: number; total: number; message: string
  isError: boolean; errorMsg: string
  onFile: (f: File) => void
  onExpand: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)
  const pick = (f: File) => { if (f.type.startsWith('image/')) onFile(f) }

  return (
    <div className="mb-7">
      <input ref={inputRef} type="file" accept="image/*" className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) pick(f) }} />

      <div
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) pick(f) }}
        className={`w-full relative rounded-xl overflow-hidden select-none transition-all
          ${drag ? 'ring-1 ring-white/30' : ''}`}
        style={{ boxShadow: '0 4px 20px rgba(0,0,0,0.55)' }}
      >
        {preview ? (
          <>
            <img
              src={preview} alt="uploaded"
              className="w-full block object-cover cursor-zoom-in"
              onClick={onExpand}
            />
            <button
              onClick={() => inputRef.current?.click()}
              className="absolute bottom-2 right-2 text-[11px] text-white/50 bg-black/40
                         hover:bg-black/60 px-2.5 py-1 rounded-lg transition-colors"
            >
              Change
            </button>
          </>
        ) : (
          <div
            onClick={() => inputRef.current?.click()}
            className="flex items-center justify-center bg-[#cbcbcb] text-[#888] text-[13px] cursor-pointer"
            style={{ minHeight: 186 }}
          >
            Upload picture
          </div>
        )}
      </div>

      {isRunning && (
        <div className="mt-3.5 space-y-2">
          <div className="flex justify-between">
            <span className="text-[12px] text-white/40">{message}</span>
            <span className="text-[12px] text-white/25 font-mono">{step}/{total}</span>
          </div>
          <div className="h-px bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-white/50 rounded-full transition-all duration-500"
              style={{ width: `${total > 0 ? (step / total) * 100 : 0}%` }} />
          </div>
        </div>
      )}
      {isError && <p className="mt-2 text-[12px] text-red-400/70">{errorMsg}</p>}
    </div>
  )
}

// ── Cloud analysis ────────────────────────────────────────────────────────────

function CloudAnalysisSection({ analysis }: { analysis: Analysis }) {
  const ct = analysis.cloud_type
  return (
    <div className="mb-7">
      <SectionTitle>Cloud analysis</SectionTitle>
      <Card>
        <p className="text-[16px] text-white font-medium leading-tight">{ct.name}</p>
        <p className="text-[12px] text-white/40 mt-0.5">{ct.level} · {ct.altitude_range}</p>
        <div className="mt-3 border-t border-white/5 pt-1">
          <KVRow label="Confidence" value={`${(ct.confidence * 100).toFixed(0)}%`} />
          {ct.top3.map(([name, prob]) => (
            <KVRow key={name} label={name} value={`${(prob * 100).toFixed(0)}%`} />
          ))}
        </div>
      </Card>
    </div>
  )
}

// ── Solar geometry ────────────────────────────────────────────────────────────

function SolarSection({ analysis }: { analysis: Analysis }) {
  const s = analysis.solar
  const rows: [string, string][] = [
    ['Time of day',   s.time_of_day.replace(/_/g, ' ')],
    ['Sun elevation', s.elevation_deg !== null ? `${s.elevation_deg}°` : 'unknown'],
    ['Hemisphere',    s.hemisphere ?? 'unknown'],
    ['Season',        s.season_hint ?? 'unknown'],
    ...(s.lat_range ? [['Estimated lat', `${s.lat_range[0]}° – ${s.lat_range[1]}°`] as [string, string]] : []),
  ]
  return (
    <div className="mb-7">
      <SectionTitle>Solar Geometry</SectionTitle>
      <Card>
        {rows.map(([label, value]) => <KVRow key={label} label={label} value={value} />)}
      </Card>
    </div>
  )
}

// ── Matches ───────────────────────────────────────────────────────────────────

function MatchesSection({ matches, onSelect, selected }: {
  matches: Match[]; onSelect: (m: Match) => void; selected: Match | null
}) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? matches : matches.slice(0, 10)

  return (
    <div className="mb-7">
      <SectionTitle>Matches</SectionTitle>
      <Card noPad>
        {visible.map((m, i) => {
          const active = selected?.id === m.id
          return (
            <button key={m.id} onClick={() => onSelect(m)}
              className={`w-full text-left flex items-center gap-3 px-5 py-3
                border-b border-white/5 last:border-0 transition-colors
                ${active ? 'bg-white/8' : 'hover:bg-white/4'}`}
            >
              <span className="text-[13px] text-white/30 font-mono w-6 shrink-0">{i + 1}.</span>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] text-white/60 font-mono tabular-nums truncate">
                  {m.lat.toFixed(2)}°, {m.lon.toFixed(2)}°
                </p>
                <p className="text-[11px] text-white/25 font-mono tabular-nums">
                  {new Date(m.captured_at).toISOString().slice(0, 10)}
                </p>
              </div>
              <span className="text-[13px] text-white/35 font-mono tabular-nums shrink-0">
                {m.scores.combined.toFixed(3)}
              </span>
            </button>
          )
        })}
        <div className="flex items-center justify-between px-5 py-3.5 border-t border-white/5">
          <span className="text-[13px] text-white/25">Coverage</span>
          {matches.length > 10 && (
            <Btn onClick={() => setExpanded(e => !e)}>
              {expanded ? 'Show less' : 'Show more'}
            </Btn>
          )}
        </div>
      </Card>
    </div>
  )
}

// ── Satellite view ────────────────────────────────────────────────────────────

function SatelliteSection({ match, overlayActive, onToggleOverlay }: {
  match: Match; overlayActive: boolean; onToggleOverlay: () => void
}) {
  const date = new Date(match.captured_at).toISOString().slice(0, 16).replace('T', ' ') + ' UTC'

  return (
    <div>
      <SectionTitle>Satellite view</SectionTitle>

      {/* Thumbnail */}
      <div className="w-full relative rounded-xl overflow-hidden bg-[#cbcbcb] mb-4"
        style={{ minHeight: 190, boxShadow: '0 4px 20px rgba(0,0,0,0.55)' }}>
        {match.thumbnail_url ? (
          <img src={match.thumbnail_url} alt="satellite" className="w-full block object-cover"
            onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
        ) : (
          <div className="flex items-center justify-center text-[#888] text-[13px]"
            style={{ minHeight: 190 }}>
            PICTURE
          </div>
        )}
        {/* Dark gradient so the button is always readable */}
        <div className="absolute bottom-0 left-0 right-0 h-14 pointer-events-none"
          style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.55), transparent)' }} />
        <div className="absolute bottom-3 right-3">
          <button
            onClick={onToggleOverlay}
            className="text-[11px] text-white/50 bg-black/40 hover:bg-black/60 px-2.5 py-1 rounded-lg transition-colors"
          >
            {overlayActive ? 'Hide' : 'View'}
          </button>
        </div>
      </div>

      {/* Info card */}
      <Card className="mb-4">
        <KVRow label="Coordinates" value={`${match.lat.toFixed(4)}°, ${match.lon.toFixed(4)}°`} />
        <KVRow label="Captured"    value={date} />
        <KVRow label="Collection"  value={match.collection} />
        <KVRow label="Cloud cover" value={`${match.cloud_cover_pct.toFixed(1)}%`} />
      </Card>

      <p className="text-[13px] text-white/35 mb-2">Score breakdown</p>
      <Card className="mb-4">
        <KVRow label="Combined" value={match.scores.combined.toFixed(3)} />
        <KVRow label="Coverage" value={match.scores.coverage.toFixed(3)} />
        <KVRow label="Visual"   value={match.scores.similarity.toFixed(3)} />
      </Card>

      <div className="mt-5 space-y-0.5 text-[11px] text-white/20">
        <p>CLIP ViT-B/32 · SegFormer B2 · LBP texture</p>
        <p>{match.collection}</p>
        <p suppressHydrationWarning>Analysis {new Date().toISOString().slice(0, 10)}</p>
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [preview,      setPreview]      = useState<string | null>(null)
  const [photoModal,   setPhotoModal]   = useState(false)
  const [status,       setStatus]       = useState<PipelineStatus>({ stage: 'idle' })
  const [analysis,     setAnalysis]     = useState<Analysis | null>(null)
  const [matches,      setMatches]      = useState<Match[]>([])
  const [selected,     setSelected]     = useState<Match | null>(null)
  const [overlayActive, setOverlayActive] = useState(false)
  const [cloudDay,      setCloudDay]      = useState(CLOUD_MAX_DAYS)
  const [cloudHour,     setCloudHour]     = useState(12)
  const [cloudsVisible, setCloudsVisible] = useState(false)

  const analysisRef      = useRef<Analysis | null>(null)
  const viewerRef        = useRef<CesiumComponentRef<Cesium.Viewer> | null>(null)
  const overlayLayerRef  = useRef<ImageryLayer | null>(null)
  const overlayActiveRef = useRef(false)
  const cloudLayerRef    = useRef<ImageryLayer | null>(null)

  const cloudDate = useMemo(() => dayToDate(cloudDay), [cloudDay])

  // Init viewer: labels + skybox
  useEffect(() => {
    let timerId: ReturnType<typeof setTimeout> | undefined
    const init = (attempts: number) => {
      const viewer = viewerRef.current?.cesiumElement
      if (viewer) {
        viewer.imageryLayers.add(LABELS_LAYER)
        if (viewer.scene.skyBox) viewer.scene.skyBox.show = false
        viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#0a0a12')
        return
      }
      if (attempts < 12) timerId = setTimeout(() => init(attempts + 1), 400)
    }
    timerId = setTimeout(() => init(0), 400)
    return () => clearTimeout(timerId)
  }, [])

  // Cloud imagery layer management
  const applyCloudLayer = useCallback((date: string, hour: number, show: boolean) => {
    const viewer = viewerRef.current?.cesiumElement
    if (!viewer) return
    if (cloudLayerRef.current) {
      viewer.imageryLayers.remove(cloudLayerRef.current, true)
      cloudLayerRef.current = null
    }
    if (!show) return
    const provider = new UrlTemplateImageryProvider({ url: gibsUrl(date, hour), maximumLevel: 9 })
    const layer = new ImageryLayer(provider, { alpha: 0.92 })
    const n = viewer.imageryLayers.length
    viewer.imageryLayers.add(layer, Math.max(0, n - 1))
    cloudLayerRef.current = layer
  }, [])

  const handleDayChange = useCallback((day: number) => {
    setCloudDay(day)
    if (cloudsVisible) applyCloudLayer(dayToDate(day), cloudHour, true)
  }, [cloudsVisible, cloudHour, applyCloudLayer])

  const handleHourChange = useCallback((hour: number) => {
    setCloudHour(hour)
    if (cloudsVisible) applyCloudLayer(cloudDate, hour, true)
  }, [cloudsVisible, cloudDate, applyCloudLayer])

  const toggleClouds = useCallback(() => {
    const next = !cloudsVisible
    setCloudsVisible(next)
    applyCloudLayer(cloudDate, cloudHour, next)
  }, [cloudsVisible, cloudDate, cloudHour, applyCloudLayer])

  const hideOverlay = useCallback(() => {
    overlayActiveRef.current = false
    setOverlayActive(false)
    const viewer = viewerRef.current?.cesiumElement
    if (viewer && overlayLayerRef.current) {
      viewer.imageryLayers.remove(overlayLayerRef.current, true)
      overlayLayerRef.current = null
    }
  }, [])

  const selectMatch = useCallback((m: Match | null) => {
    hideOverlay()
    setSelected(m)
  }, [hideOverlay])

  const handleFile = useCallback(async (file: File) => {
    setPreview(URL.createObjectURL(file))
    setStatus({ stage: 'running', step: 0, total: 5, message: 'Starting...' })
    setAnalysis(null); setMatches([]); selectMatch(null)
    await analyzeImage(
      file,
      (step, total, message) => setStatus({ stage: 'running', step, total, message }),
      (a)  => { setAnalysis(a); analysisRef.current = a },
      (ms) => {
        setMatches(ms)
        setStatus({ stage: 'done', analysis: analysisRef.current!, matches: ms })
        if (ms.length > 0) selectMatch(ms[0])
      },
      (msg) => setStatus({ stage: 'error', message: msg }),
    )
  }, [selectMatch])

  const toggleOverlay = useCallback(async () => {
    const viewer = viewerRef.current?.cesiumElement
    if (!viewer) return
    if (overlayActiveRef.current) { hideOverlay(); return }
    if (!selected?.thumbnail_url || !selected.bbox) return
    const { west, south, east, north } = selected.bbox
    const rect = Rectangle.fromDegrees(west, south, east, north)
    overlayActiveRef.current = true
    setOverlayActive(true)
    viewer.camera.flyTo({ destination: rect, duration: 1.5 })
    try {
      const provider = await SingleTileImageryProvider.fromUrl(selected.thumbnail_url, { rectangle: rect })
      if (!overlayActiveRef.current) return
      const layer = new ImageryLayer(provider, { alpha: 0.8 })
      viewer.imageryLayers.add(layer)
      overlayLayerRef.current = layer
    } catch {
      if (overlayActiveRef.current) { overlayActiveRef.current = false; setOverlayActive(false) }
    }
  }, [selected, hideOverlay])

  const step    = status.stage === 'running' ? status.step    : 0
  const total   = status.stage === 'running' ? status.total   : 5
  const message = (status.stage === 'running' || status.stage === 'error') ? status.message : ''
  const flyDest = useMemo(
    () => selected ? Cartesian3.fromDegrees(selected.lon, selected.lat - 3, 2_200_000) : undefined,
    [selected?.id]  // only recompute when the selected match actually changes
  )

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>

      {/* Globe */}
      <Viewer ref={viewerRef} full baseLayer={BASE_LAYER}
        timeline={false} animation={false} baseLayerPicker={false}
        navigationHelpButton={false} homeButton={false} geocoder={false}
        sceneModePicker={false} selectionIndicator={false} infoBox={false}
      >
        {flyDest && <CameraFlyTo destination={flyDest} duration={1.8} />}
        {matches.slice(0, 10).map((m, i) => (
          <Entity key={m.id}
            position={Cartesian3.fromDegrees(m.lon, m.lat)}
            onClick={() => selectMatch(m)}
            billboard={{
              image: PUSHPIN,
              scale: selected?.id === m.id ? 1.35 : i === 0 ? 1.1 : 0.85,
              verticalOrigin: VerticalOrigin.CENTER,
              horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
              heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
            }}
            label={i === 0 ? {
              text: `#1  ${m.scores.combined.toFixed(3)}`,
              font: '11px monospace',
              fillColor: Color.WHITE,
              outlineColor: Color.BLACK,
              outlineWidth: 2,
              pixelOffset: new Cartesian2(0, -40),
              showBackground: true,
              backgroundColor: Color.fromCssColorString('#111111').withAlpha(0.9),
              backgroundPadding: new Cartesian2(7, 4),
            } : undefined}
          />
        ))}
      </Viewer>

      {/* Cloud controls — centered over globe */}
      <div className="absolute bottom-7 left-[22rem] right-0 flex justify-center pointer-events-none z-10">
        <div style={{ pointerEvents: 'auto' }}>
          <CloudControls
            day={cloudDay} onDay={handleDayChange}
            hour={cloudHour} onHour={handleHourChange}
            visible={cloudsVisible} onToggle={toggleClouds}
          />
        </div>
      </div>

      {/* Sidebar */}
      <aside
        className="absolute top-0 left-0 h-full w-[22rem] bg-[#1a1a1a]
                   border-r border-white/8 overflow-y-auto pointer-events-auto"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.15) transparent', padding: '14px 14px 14px' }}
      >
        <h1 className="text-center text-[18px] font-medium text-white mt-0 mb-4">
          IRTC - I remember that cloud
        </h1>

        <PhotoArea
          preview={preview}
          isRunning={status.stage === 'running'}
          step={step} total={total} message={message}
          isError={status.stage === 'error'} errorMsg={message}
          onFile={handleFile}
          onExpand={() => preview && setPhotoModal(true)}
        />

        {analysis && <CloudAnalysisSection analysis={analysis} />}
        {analysis && <SolarSection analysis={analysis} />}
        {matches.length > 0 && (
          <MatchesSection matches={matches} onSelect={selectMatch} selected={selected} />
        )}
        {selected && (
          <SatelliteSection match={selected} overlayActive={overlayActive} onToggleOverlay={toggleOverlay} />
        )}
      </aside>

      {/* Photo modal */}
      {photoModal && preview && (
        <PhotoModal src={preview} onClose={() => setPhotoModal(false)} />
      )}
    </div>
  )
}
