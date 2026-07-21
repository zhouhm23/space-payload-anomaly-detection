/**
 * 全局系统状态 store
 *
 * - 后端启动状态（startupStatus）：idle/initializing/ready/failed
 * - 系统信息（systemInfo）：顶栏用（天地链接、UTC、系统名）
 * - 设备树（deviceTree + health）：左栏 + 中央轮播用
 *
 * 1b 阶段先存这些，后续按需扩展。
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api } from '@/api'

export type StartupState = 'idle' | 'initializing' | 'ready' | 'failed' | 'disconnected'

export interface SystemInfo {
  system_title: string
  utc_time: string
  utc_epoch: number
  link_status: 'online' | 'degraded' | 'offline' | 'waiting' | 'initializing' | 'error'
  link_latency_ms: number | null
  note?: string
}

export interface DeviceNode {
  id: string
  name: string
  type: 'folder' | 'sensor'
  sourceId?: string
  channelName?: string
  blockSize?: number
  yMin?: number
  yMax?: number
  unit?: string
  threshold?: number
  description?: string
  isSpecial?: boolean
  children?: DeviceNode[]
  [key: string]: unknown
}

export interface DeviceTreeData {
  device_tree: DeviceNode[]
  aggregation_strategy: 'min' | 'mean'
  health: {
    system: number
    threshold: number
    channels: Record<string, number>
    folders?: Record<string, { min: number; mean: number }>
  }
}

export const useSystemStore = defineStore('system', () => {
  const startupState = ref<StartupState>('initializing')
  const startupError = ref<string>('')
  const systemInfo = ref<SystemInfo | null>(null)
  const deviceTree = ref<DeviceTreeData | null>(null)
  // RUL 退化预测数据（特殊传感器用）
  const rulData = ref<Array<{
    channel: string
    rul: number
    max_rul: number
    unit: string
    model: string
    history?: number[]
  }>>([])
  // 当前选中通道（轮播 + 手动切换共享）
  const currentChannel = ref<string>('')
  // 当前通道序号 / 总通道数（用于顶栏显示 X/Y）
  const carouselChannels = ref<string[]>([]) // 参与轮播的通道列表
  const carouselIndex = ref(0)

  /** 检查启动状态 */
  async function checkStartup() {
    try {
      const res = await api.startupStatus()
      startupState.value = res.state as StartupState
      startupError.value = res.error || ''
    } catch (e) {
      startupState.value = 'disconnected'
      startupError.value = (e as Error).message
    }
  }

  /** 刷新系统信息 */
  async function refreshSystemInfo() {
    try {
      systemInfo.value = await api.systemInfo()
    } catch (e) {
      // 静默失败（顶栏会保留上次数据）
      console.warn('[systemInfo] refresh failed:', e)
    }
  }

  /** 刷新设备树 */
  async function refreshDeviceTree() {
    try {
      deviceTree.value = await api.deviceTree()
      // 更新参与轮播的通道列表（仅非特殊的一维数据源传感器）
      const sensors: string[] = []
      function walk(nodes: DeviceNode[]) {
        for (const n of nodes) {
          if (n.type === 'sensor' && !n.isSpecial && n.channelName) {
            sensors.push(n.channelName)
          } else if (n.type === 'folder' && n.children) {
            walk(n.children)
          }
        }
      }
      if (deviceTree.value) walk(deviceTree.value.device_tree)
      carouselChannels.value = sensors
      // 若当前通道被清空了，选第一个
      if (!currentChannel.value && sensors.length > 0) {
        currentChannel.value = sensors[0]
        carouselIndex.value = 0
      }
    } catch (e) {
      console.warn('[deviceTree] refresh failed:', e)
    }
  }

  /** 刷新 RUL 退化预测（5s，特殊传感器用） */
  async function refreshRul() {
    try {
      const res = await api.rul()
      rulData.value = res.channels || []
    } catch (e) {
      // RUL 服务未启用时静默（保留上次数据）
    }
  }

  /** 按通道名取 RUL 数据 */
  function getRul(channel: string) {
    return rulData.value.find((r) => r.channel === channel) || null
  }

  /** 切到下一个轮播通道 */
  function nextCarousel() {
    if (carouselChannels.value.length === 0) return
    carouselIndex.value = (carouselIndex.value + 1) % carouselChannels.value.length
    currentChannel.value = carouselChannels.value[carouselIndex.value]
  }

  /** 手动选中通道（重置轮播序号） */
  function selectChannel(ch: string) {
    currentChannel.value = ch
    const idx = carouselChannels.value.indexOf(ch)
    if (idx >= 0) carouselIndex.value = idx
  }

  // ── 显示名映射 ────────────────────────────────────────────────────────────
  // 内部 channelName → 显示 name（如 VS-sine → S1）
  // 修改设备树后，历史告警里的 channel 会自动用新名显示（实时映射，无需数据迁移）
  const displayNameMap = computed<Record<string, string>>(() => {
    const map: Record<string, string> = {}
    function walk(nodes: DeviceNode[]) {
      for (const n of nodes || []) {
        if (n.type === 'sensor' && n.channelName && n.name) {
          map[n.channelName] = n.name
        }
        if (n.children) walk(n.children)
      }
    }
    if (deviceTree.value) walk(deviceTree.value.device_tree)
    return map
  })

  /** 内部 channelName → 显示名（找不到时返回原值） */
  function displayName(channel: string | undefined | null): string {
    if (!channel) return '—'
    return displayNameMap.value[channel] || channel
  }

  // ── 自动轮播 ────────────────────────────────────────────────────────────
  // 默认 15 秒切换（需求书要求），间隔可从 theme.carousel.intervalMs 读
  let carouselTimer: ReturnType<typeof setInterval> | null = null
  const carouselIntervalMs = ref<number>(15000)

  /** 启动自动轮播（如已启动则忽略）。 */
  function startCarousel(intervalMs?: number) {
    if (intervalMs) carouselIntervalMs.value = intervalMs
    if (carouselTimer) return
    carouselTimer = setInterval(() => {
      if (carouselChannels.value.length > 0) {
        nextCarousel()
      }
    }, carouselIntervalMs.value)
  }

  /** 停止自动轮播。 */
  function stopCarousel() {
    if (carouselTimer) {
      clearInterval(carouselTimer)
      carouselTimer = null
    }
  }

  return {
    startupState,
    startupError,
    systemInfo,
    deviceTree,
    currentChannel,
    carouselChannels,
    carouselIndex,
    carouselIntervalMs,
    checkStartup,
    refreshSystemInfo,
    refreshDeviceTree,
    refreshRul,
    getRul,
    nextCarousel,
    selectChannel,
    startCarousel,
    stopCarousel,
    displayName,
  }
})
