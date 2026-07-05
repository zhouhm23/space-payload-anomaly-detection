/**
 * HTTP client — thin fetch wrappers for the 8 FastAPI endpoints.
 *
 * During dev Vite proxies ``/api`` to ``localhost:8501``.  In production
 * the built dist is served by the same FastAPI process so relative URLs
 * work directly.
 */

import type {
  PollResponse,
  ForecastResponse,
  DeviceTreeConfig,
  HealthResponse,
  AlertsResponse,
  WarningsResponse,
  SensorsResponse,
  PredictScoresResponse,
} from './types'

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${url} → ${r.status}`)
  return r.json() as Promise<T>
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url} → ${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  poll(sourceId: string, sampleRate = 50.0, blockSize = 512): Promise<PollResponse> {
    return postJson('/api/poll', { source_id: sourceId, sample_rate: sampleRate, block_size: blockSize })
  },

  forecast(values: number[]): Promise<ForecastResponse> {
    return postJson('/api/forecast', { values })
  },

  getConfig(): Promise<DeviceTreeConfig> {
    return getJson(`/api/config?t=${Date.now()}`)
  },

  saveConfig(tree: DeviceTreeConfig): Promise<{ status: string }> {
    return postJson('/api/config', tree)
  },

  reset(): Promise<{ status: string }> {
    return postJson('/api/reset', {})
  },

  health(): Promise<HealthResponse> {
    return getJson('/api/health')
  },

  alerts(): Promise<AlertsResponse> {
    return getJson('/api/alerts')
  },

  warnings(): Promise<WarningsResponse> {
    return getJson('/api/warnings')
  },

  sensors(): Promise<SensorsResponse> {
    return getJson('/api/sensors')
  },

  predictScores(channel: string): Promise<PredictScoresResponse> {
    return getJson(`/api/predict-scores?channel=${encodeURIComponent(channel)}`)
  },
}
