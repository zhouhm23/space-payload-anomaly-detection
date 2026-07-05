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

export interface DeviceNode {
  id: string
  name: string
  description?: string
  type?: 'sensor' | 'folder'
  sourceId?: string
  channelName?: string
  blockSize?: number
  children?: DeviceNode[]
}

export interface DeviceTreeConfig {
  device_tree: DeviceNode[]
}

export interface HealthResponse {
  system: number
  channels: Record<string, number>
  threshold: number
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
