/**
 * useForecast — TTM-R3 prediction request + per-block cache.
 *
 * Mirrors the legacy ``computePredict`` + ``lastForecastBlockIdx`` /
 * ``cachedPredTele`` / ``cachedPredScores`` cache variables.
 */

import { ref } from 'vue'
import { api } from '@/api/client'

export interface ForecastResult {
  predValues: number[]
}

export function useForecast() {
  const lastForecastBlockIdx = ref(-1)

  async function computePredict(
    telemetry: number[][],
    blockIdx: number,
  ): Promise<{ predValues: number[] } | null> {
    if (!telemetry || telemetry.length < 10) return { predValues: [] }
    // telemetry is [sample_index, value]
    const values = telemetry.slice(-512).map((p) => p[1])
    try {
      const data = await api.forecast(values)
      if (data.error || !data.prediction) return { predValues: [] }
      // Return raw prediction values; caller aligns them to timestamps.
      return { predValues: data.prediction }
    } catch (e) {
      console.warn('TTM-R3 预测失败:', e)
      return null
    }
  }

  function invalidate(): void {
    lastForecastBlockIdx.value = -1
  }

  return { lastForecastBlockIdx, computePredict, invalidate }
}
