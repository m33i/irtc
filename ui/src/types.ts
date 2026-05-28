export interface CloudType {
  name: string
  abbr: string
  level: string
  altitude_range: string
  confidence: number
  top3: [string, number][]
}

export interface SolarInfo {
  sun_visible: boolean
  elevation_deg: number | null
  time_of_day: string
  hour_range: [number, number]
  hemisphere: string | null
  lat_range: [number, number] | null
  season_hint: string | null
  confidence: number
}

export interface Features {
  cloud_coverage_pct: number
  dominant_brightness: number
  embedding_dim: number
}

export interface SearchConstraints {
  lat_range: [number, number] | null
  hour_range: [number, number] | null
  season_hint: string | null
  hemisphere: string | null
  cloud_coverage: [number, number]
  cloud_level: string
}

export interface Analysis {
  image_path: string
  sky_ratio: number
  cloud_type: CloudType
  solar: SolarInfo
  features: Features
  search_constraints: SearchConstraints
}

export interface MatchScores {
  similarity: number
  coverage: number
  combined: number
}

export interface MatchBBox {
  west: number; south: number; east: number; north: number
}

export interface Match {
  rank: number
  id: string
  collection: string
  lat: number
  lon: number
  captured_at: string
  cloud_cover_pct: number
  thumbnail_url: string | null
  bbox: MatchBBox
  scores: MatchScores
}

export type PipelineStatus =
  | { stage: 'idle' }
  | { stage: 'uploading' }
  | { stage: 'running'; step: number; total: number; message: string }
  | { stage: 'done'; analysis: Analysis; matches: Match[] }
  | { stage: 'error'; message: string }
