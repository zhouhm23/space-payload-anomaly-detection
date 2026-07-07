/**
 * Telemetry window store — SQLite-backed scrolling window viewer.
 *
 * Replaces the old block-based usePoll + telemetry store for the chart
 * panel.  The frontend only does "fetch latest N points → draw".  All
 * business logic (poll, detect, forecast) runs in the backend auto-poll
 * thread.
 *
 * Three modes:
 *  - realtime:  auto-scroll, right-edge = DB latest, poll every 2s
 *  - frozen:    stop scrolling, poll continues in backend, user can
 *               drag/jump and change window length
 *  - reset:     jump to latest but NOT realtime (user must click 实时)
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api } from '@/api/client'
import type { WindowResponse, WindowRawPoint, PredictionBatch } from '@/api/types'

export type ViewMode = 'realtime' | 'frozen' | 'reset'

export const DEFAULT_WINDOW = 512
export const MIN_WINDOW = 100
export const MAX_WINDOW = 1000
export const POLL_INTERVAL_MS = 2000

export const useTelemetryWindowStore = defineStore('telemetryWindow', () => {
  const mode = ref<ViewMode>('frozen')
  const channel = ref<string>('')
  const windowSize = ref(DEFAULT_WINDOW)
  /** Right-edge timestamp (epoch seconds). null = use DB latest. */
  const viewEndTs = ref<number | null>(null)

  const raw = ref<WindowRawPoint[]>([])
  const predictions = ref<PredictionBatch[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const lastFetchAt = ref(0)

  let pollTimer: ReturnType<typeof setInterval> | null = null

  // ---- computed: chart-ready series ----

  /**
   * Sort an array of [ts_ms, value] by ts_ms ascending and deduplicate
   * by ts_ms (keeping the last occurrence).  ECharts ``type:'time'``
   * requires monotonically ascending x-values; duplicate or
   * out-of-order timestamps produce vertical spikes (visual zig-zag).
   */
  function sortAndDedup(points: [number, number][]): [number, number][] {
    if (points.length <= 1) return points
    const sorted = [...points].sort((a, b) => a[0] - b[0])
    const map = new Map<number, number>()
    for (const [ts, v] of sorted) map.set(ts, v)
    return [...map.entries()]
  }

  /** Telemetry series: [ts_ms, value], sorted + deduped */
  const teleSeries = computed<[number, number][]>(() =>
    sortAndDedup(
      raw.value
        .filter((p) => p.raw !== null)
        .map((p) => [p.received_at * 1000, p.raw as number]),
    ),
  )

  /** Anomaly score series: [ts_ms, score], sorted + deduped */
  const scoreSeries = computed<[number, number][]>(() =>
    sortAndDedup(
      raw.value
        .filter((p) => p.score !== null)
        .map((p) => [p.received_at * 1000, p.score as number]),
    ),
  )

  /**
   * Predicted values series, expanded into [ts_ms, value] points.
   * No bridging point — the dashed line starts cleanly from
   * predict_start, separate from the solid raw line.
   */
  const predTeleSeries = computed<[number, number][]>(() => {
    const result: [number, number][] = []
    for (const batch of predictions.value) {
      const n = batch.prediction.length
      if (n === 0) continue
      const startMs = batch.predict_start * 1000
      const endMs = batch.predict_end * 1000
      const step = n > 1 ? (endMs - startMs) / (n - 1) : 0
      for (let i = 0; i < n; i++) {
        const ts = startMs + i * step
        const v = batch.prediction[i]
        if (v !== null && v !== undefined) {
          result.push([ts, v])
        }
      }
    }
    return result
  })

  /** Predicted anomaly score series: [ts_ms, score] */
  const predScoreSeries = computed<[number, number][]>(() => {
    const result: [number, number][] = []
    for (const batch of predictions.value) {
      const scores = batch.predict_scores
      const n = scores.length
      if (n === 0) continue
      const startMs = batch.predict_start * 1000
      const endMs = batch.predict_end * 1000
      const step = n > 1 ? (endMs - startMs) / (n - 1) : 0
      for (let i = 0; i < n; i++) {
        const ts = startMs + i * step
        const v = scores[i]
        if (v !== null && v !== undefined) {
          result.push([ts, v])
        }
      }
    }
    return result
  })

  // ---- actions ----

  async function fetchWindow(): Promise<void> {
    if (!channel.value || loading.value) return
    loading.value = true
    error.value = null
    try {
      // In realtime mode, always use endTs=null (DB latest)
      const endTs = mode.value === 'realtime' ? undefined : viewEndTs.value ?? undefined
      const resp: WindowResponse = await api.window(channel.value, windowSize.value, endTs)
      raw.value = resp.raw
      predictions.value = resp.predictions
      // Track the actual right edge so frozen mode stays put
      if (mode.value === 'realtime' && resp.end_ts) {
        viewEndTs.value = resp.end_ts
      }
      lastFetchAt.value = Date.now()
    } catch (e) {
      error.value = String(e)
    } finally {
      loading.value = false
    }
  }

  function startRealtime(): void {
    mode.value = 'realtime'
    viewEndTs.value = null
    stopPoll()
    pollTimer = setInterval(() => {
      fetchWindow()
    }, POLL_INTERVAL_MS)
    fetchWindow()
  }

  function freeze(): void {
    mode.value = 'frozen'
    stopPoll()
    // viewEndTs stays at current position — frozen for drag/jump
  }

  function reset(): void {
    mode.value = 'reset'
    viewEndTs.value = null
    stopPoll()
    fetchWindow()
  }

  /** Jump to a specific right-edge timestamp (frozen mode only). */
  function jumpTo(ts: number): void {
    if (mode.value === 'realtime') return
    viewEndTs.value = ts
    mode.value = 'frozen'
    fetchWindow()
  }

  /** Pan the window to a new right-edge (frozen mode drag). */
  function panTo(newEndTsMs: number): void {
    if (mode.value === 'realtime') return
    viewEndTs.value = newEndTsMs / 1000
    mode.value = 'frozen'
    fetchWindow()
  }

  /** Change window length (frozen mode only). */
  function setWindowSize(n: number): void {
    windowSize.value = Math.max(MIN_WINDOW, Math.min(MAX_WINDOW, n))
    if (mode.value !== 'realtime') {
      fetchWindow()
    }
  }

  function setChannel(ch: string): void {
    const wasRealtime = mode.value === 'realtime'
    channel.value = ch
    viewEndTs.value = null
    if (wasRealtime) {
      // Keep realtime mode going — just switch the data source
      fetchWindow()
    } else {
      mode.value = 'frozen'
      stopPoll()
      fetchWindow()
    }
  }

  function stopPoll(): void {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    // state
    mode,
    channel,
    windowSize,
    viewEndTs,
    raw,
    predictions,
    loading,
    error,
    lastFetchAt,
    // computed
    teleSeries,
    scoreSeries,
    predTeleSeries,
    predScoreSeries,
    // actions
    fetchWindow,
    startRealtime,
    freeze,
    reset,
    jumpTo,
    panTo,
    setWindowSize,
    setChannel,
    stopPoll,
  }
})
