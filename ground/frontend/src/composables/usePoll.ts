/**
 * 轮询 composable
 *
 * 用法：
 *   const { data, error, isReady } = usePoll(() => api.alerts(20), 3000, { autoStart: true })
 *
 * 设计要点：
 * - 支持 start/stop/pause/resume
 * - 后端 503（未就绪）时降速重试（避免空转打爆 CPU）
 * - 组件卸载自动清理
 * - 请求并发去重（同一 fetcher 不会并发）
 */
import { ref, onUnmounted, type Ref } from 'vue'
import { BackendNotReadyError } from '@/api'

interface PollOptions<T> {
  /** 立即触发一次 */
  immediate?: boolean
  /** 自动启动定时器 */
  autoStart?: boolean
  /** 后端未就绪时的降速间隔（默认 3s） */
  notReadyInterval?: number
  /** 错误回调（不影响下次轮询） */
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
  const isReady = ref(false) // 数据首次成功拉到后置 true
  const failCount = ref(0)

  let timer: ReturnType<typeof setInterval> | null = null
  let inflight: Promise<T | null> | null = null

  async function tick(): Promise<T | null> {
    // 并发去重：上一次还没回来就跳过
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
        // 后端未就绪 → 暂时降速（重新调度）
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

  /** 暂时暂停（如模态打开时） */
  function pause() {
    stop()
  }

  /** 恢复（如模态关闭），并立即触发一次 */
  function resume() {
    tick()
    start()
  }

  onUnmounted(() => stop())

  if (immediate) tick()
  if (autoStart) start()

  return { data, error, loading, isReady, failCount, tick, start, stop, pause, resume }
}
