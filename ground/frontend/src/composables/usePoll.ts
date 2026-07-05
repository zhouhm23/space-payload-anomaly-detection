/**
 * usePoll — polling scheduler composable.
 *
 * Replicates the legacy ``schedulePoll`` / ``togglePlayPause`` / ``resetStream``
 * control flow.  Polling fires every ``POLL_INTERVAL_S`` seconds while
 * ``playing`` is true.  The health/alerts/warnings stores are refreshed
 * after each successful poll so the dashboard stays live.
 */

import { ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { useHealthStore } from '@/stores/health'
import { useLinkStore } from '@/stores/link'

export const POLL_INTERVAL_S = 2
export const DEFAULT_SAMPLE_RATE = 50.0
export const DEFAULT_BLOCK_SIZE = 1024

export function usePoll() {
  const telemetry = useTelemetryStore()
  const tree = useDeviceTreeStore()
  const health = useHealthStore()
  const link = useLinkStore()

  const timer = ref<ReturnType<typeof setTimeout> | null>(null)

  async function fetchBlock(blockSizeOverride?: number): Promise<boolean> {
    const sid = tree.selectedSourceId() || tree.firstSensorSourceId()
    if (!sid) return false
    const blockSize = blockSizeOverride ?? tree.selectedBlockSize(DEFAULT_BLOCK_SIZE)
    const ok = await telemetry.pollOnce(sid, DEFAULT_SAMPLE_RATE, blockSize)
    if (ok) {
      // refresh PHM side-panels after new data
      await health.refreshAll().catch(() => {})
    }
    link.updateFromPolling(telemetry.playing, timer.value !== null, telemetry.blocks.length > 0)
    return ok
  }

  function schedulePoll(): void {
    if (!telemetry.playing) return
    timer.value = setTimeout(async () => {
      await fetchBlock()
      if (telemetry.playing) schedulePoll()
    }, POLL_INTERVAL_S * 1000)
  }

  function togglePlayPause(): void {
    if (telemetry.playing) {
      telemetry.playing = false
      if (timer.value) {
        clearTimeout(timer.value)
        timer.value = null
      }
    } else {
      if (!tree.anySensorInTree()) {
        alert('设备树中没有设置数据源的传感器，请先添加传感器并配置数据源')
        return
      }
      telemetry.playing = true
      schedulePoll()
    }
    link.updateFromPolling(telemetry.playing, timer.value !== null, telemetry.blocks.length > 0)
  }

  function resetStream(): void {
    if (telemetry.playing) return
    if (timer.value) {
      clearTimeout(timer.value)
      timer.value = null
    }
    telemetry.reset()
    link.updateFromPolling(false, false, false)
  }

  return {
    timer,
    fetchBlock,
    schedulePoll,
    togglePlayPause,
    resetStream,
  }
}
