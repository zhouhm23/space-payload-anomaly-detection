/**
 * PHM backend API client wrapper.
 *
 * All calls go through /api/v2/ (the DRF v2 surface) and are proxied to
 * Django :8501 via the vite dev-server proxy. Error handling: a 503
 * (Container not ready) is thrown as a dedicated error so the caller can
 * decide whether to retry.
 */
import axios from 'axios'

const client = axios.create({
  baseURL: '/api/v2/',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

// Unified error translation: 503 → backend still initialising, retryable
export class BackendNotReadyError extends Error {
  constructor(public state: string, message: string) {
    super(message)
    this.name = 'BackendNotReadyError'
  }
}

client.interceptors.response.use(
  (resp) => resp,
  (error) => {
    if (error.response?.status === 503) {
      const detail = error.response.data?.detail || '后端初始化中'
      const state = error.response.data?.state || 'unknown'
      return Promise.reject(new BackendNotReadyError(state, detail))
    }
    return Promise.reject(error)
  }
)

// ── System endpoints ───────────────────────────────────────────────────────
export const api = {
  /** Health probe (pure Django process). */
  ping: () => client.get('/ping/').then((r) => r.data),

  /** Startup-state probe. */
  startupStatus: () => client.get('/startup-status/').then((r) => r.data),

  /** Frontend theme. */
  theme: () => client.get('/theme/').then((r) => r.data),

  /** Top-bar system info. */
  systemInfo: () => client.get('/system-info/').then((r) => r.data),

  /** Device tree + health aggregation. */
  deviceTree: () => client.get('/device-tree/').then((r) => r.data),

  /** Live telemetry window (ring buffer + pred merged, 2s refresh). */
  window: (channel: string, count = 512) =>
    client
      .get('/window/', { params: { channel, count } })
      .then((r) => r.data),

  /** All-channel alert point map (red = measured, yellow = predicted). */
  alertPoints: () => client.get('/alert-points/').then((r) => r.data),

  /** Measured alert list. */
  alerts: (limit = 20) =>
    client.get('/alerts/', { params: { limit } }).then((r) => r.data),

  /** Predicted warning list. */
  warnings: (limit = 20) =>
    client.get('/warnings/', { params: { limit } }).then((r) => r.data),

  /** RUL degradation prediction. */
  rul: (channel?: string) =>
    client.get('/rul/', { params: channel ? { channel } : {} }).then((r) => r.data),
}

export type Api = typeof api
