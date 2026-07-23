/**
 * Polling composable.
 *
 * Usage:
 *   const { data, error, isReady } = usePoll(() => api.alerts(20), 3000, { autoStart: true })
 *
 * Design notes:
 * - Supports start/stop/pause/resume.
 * - On a 503 (backend not ready) it backs off (avoids busy-looping the CPU).
 * - Cleans itself up on component unmount.
 * - De-duplicates concurrent requests (a fetcher never runs twice in parallel).
 */
import { ref, onUnmounted, type Ref } from 'vue'
import { BackendNotReadyError } from '@/api'

interface PollOptions<T> {
  /** Fire once immediately. */
  immediate?: boolean
  /** Start the interval timer automatically. */
  autoStart?: boolean
  /** Back-off interval while the backend is not ready (default 3s). */
  notReadyInterval?: number
  /** Error callback (does not stop the next poll). */
  onError?: (err: unknown) => void
}

export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  options: PollOptions<T> = {}
) {
  const { immediate = false, autoStart = true, notReadyInterval = 3000, onError } = options

  const data: Ref<T | null> = ref(null)
  const error: Ref<unknown> = ref(null)
  const loading = ref(false)
  const isReady = ref(false) // flipped true after the first successful fetch
  const failCount = ref(0)

  let timer: ReturnType<typeof setInterval> | null = null
  let inflight: Promise<T | null> | null = null

  async function tick(): Promise<T | null> {
    // De-duplicate: if the previous call is still pending, skip
    if (inflight) return inflight
    loading.value = true
    inflight = fetcher()
      .then((result) => {
        data.value = result
        error.value = null
        isReady.value = true
        failCount.value = 0
        return result
      })
      .catch((err) => {
        error.value = err
        failCount.value++
        onError?.(err)
        // Backend not ready → back off (reschedule at the slower interval)
        if (err instanceof BackendNotReadyError) {
          restartTimer(notReadyInterval)
        }
        return null
      })
      .finally(() => {
        loading.value = false
        inflight = null
      })
    return inflight
  }

  function start() {
    if (timer) return
    timer = setInterval(tick, intervalMs)
  }

  function stop() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  function restartTimer(newInterval: number) {
    stop()
    timer = setInterval(tick, newInterval)
  }

  /** Pause temporarily (e.g. while a modal is open). */
  function pause() {
    stop()
  }

  /** Resume (e.g. on modal close) and fire once immediately. */
  function resume() {
    tick()
    start()
  }

  onUnmounted(() => stop())

  if (immediate) tick()
  if (autoStart) start()

  return { data, error, loading, isReady, failCount, tick, start, stop, pause, resume }
}
