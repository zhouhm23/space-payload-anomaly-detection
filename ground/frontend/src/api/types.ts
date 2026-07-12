/** Shared TypeScript types — mirror the FastAPI response contracts. */

export interface TelemetryPoint {
  /** [timestamp_ms, value] */
  0: number
  1: number
}
export type TelemetrySeries = number[][]

export interface ChannelData {
  telemetry: TelemetrySeries
  scores: TelemetrySeries
}

export interface PollResponse {
  channels: Record<string, ChannelData>
  alerts: AlertItem[]
  exhausted: boolean
  total: number
  block_size: number
}

export interface AlertItem {
  channel: string
  score: number
  step?: number
  message: string
  time: number
  type?: string
}

export interface ForecastResponse {
  context?: number[]
  prediction?: number[]
  model?: string
  error?: string
}

export interface PredictScoresResponse {
  timestamps: number[]
  scores: number[]
  predict_start: number
  predict_end: number
}

/** Normalised [0,1] position of a sensor within its module region on the
 * device diagram. ``module`` is the physical module name (e.g. "电源模块");
 * sensors without ``module`` fall back to an "未分组" region. */
export interface SensorPosition {
  module?: string
  /** 0..1 horizontal fraction within the module region */
  x?: number
  /** 0..1 vertical fraction within the module region */
  y?: number
}

export interface DeviceNode {
  id: string
  name: string
  description?: string
  type?: 'sensor' | 'folder'
  sourceId?: string
  channelName?: string
  blockSize?: number
  /** Sensor-only: diagram placement. Absent → auto-layout fallback. */
  position?: SensorPosition
  children?: DeviceNode[]
}

export interface DeviceTreeConfig {
  device_tree: DeviceNode[]
  /** Folder-health aggregation: 'min' (default, worst sensor wins) | 'mean'. */
  aggregation_strategy?: 'min' | 'mean'
}

/** Folder-level health entry returned by /api/health (when folders exist). */
export interface FolderHealth {
  name: string
  health: number
  strategy: 'min' | 'mean'
  channels: string[]
}

export interface HealthResponse {
  system: number
  channels: Record<string, number>
  threshold: number
  /** Present only when the backend has a config with folders (Slice 0+). */
  folders?: Record<string, FolderHealth>
}

export interface AlertsResponse {
  alerts: AlertItem[]
  threshold: number
}

export type WarningStatus = 'pending' | 'confirmed' | 'false'

export interface WarningItem {
  channel: string
  predict_start: number
  predict_end: number
  max_predict_score: number
  created_at: number
  status: WarningStatus
  verified_max_score: number | null
  verified_at: number | null
  message: string
  type: string
}

export interface WarningsResponse {
  warnings: WarningItem[]
}

export interface SensorItem {
  channel: string
  latest_raw: number | null
  latest_score: number
  points: number
  received_at: number | null
  health: number
}

export interface SensorsResponse {
  sensors: SensorItem[]
  system_health: number
}

/** One raw telemetry point from /api/window */
export interface WindowRawPoint {
  raw: number | null
  score: number | null
  received_at: number // epoch seconds
}

/** A predicted-values batch from /api/window */
export interface PredictionBatch {
  origin_ts: number
  predict_start: number
  predict_end: number
  prediction: number[]
  predict_scores: number[]
  model: string | null
}

/** Response from GET /api/window */
export interface WindowResponse {
  channel: string
  count: number
  end_ts: number | null
  start_ts: number | null
  raw: WindowRawPoint[]
  predictions: PredictionBatch[]
}

/** DB statistics from GET /api/db-stats */
export interface DbStatsResponse {
  enabled: boolean
  db_path?: string
  raw_telemetry?: number
  detection_results?: number
  alert_records?: number
  predictions?: number
  queue_pending?: number
}
