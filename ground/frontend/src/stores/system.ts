/**
 * Global system state store.
 *
 * - Backend startup state (startupStatus): idle/initializing/ready/failed
 * - System info (systemInfo): for the top bar (space-ground link, UTC, system name)
 * - Device tree (deviceTree + health): for the left panel + center carousel
 *
 * Phase 1b stores just these; extend as needed later.
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
  // RUL degradation prediction data (for special sensors)
  const rulData = ref<Array<{
    channel: string
    rul: number
    max_rul: number
    unit: string
    model: string
    history?: number[]
  }>>([])
  // Currently selected channel (shared by carousel + manual selection)
  const currentChannel = ref<string>('')
  // Current channel index / total (for the top-bar X/Y indicator)
  const carouselChannels = ref<string[]>([]) // channels participating in the carousel
  const carouselIndex = ref(0)

  /** Check the startup state. */
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

  /** Refresh system info. */
  async function refreshSystemInfo() {
    try {
      systemInfo.value = await api.systemInfo()
    } catch (e) {
      // Fail silently (the top bar keeps the last values)
      console.warn('[systemInfo] refresh failed:', e)
    }
  }

  /** Refresh the device tree. */
  async function refreshDeviceTree() {
    try {
      deviceTree.value = await api.deviceTree()
      // Rebuild the carousel channel list (only non-special 1-D source sensors)
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
      // If the current channel was cleared, pick the first one
      if (!currentChannel.value && sensors.length > 0) {
        currentChannel.value = sensors[0]
        carouselIndex.value = 0
      }
    } catch (e) {
      console.warn('[deviceTree] refresh failed:', e)
    }
  }

  /** Refresh RUL degradation prediction (every 5s, for special sensors). */
  async function refreshRul() {
    try {
      const res = await api.rul()
      rulData.value = res.channels || []
    } catch (e) {
      // Silent when the RUL service is disabled (keep the last values)
    }
  }

  /** Get RUL data by channel name. */
  function getRul(channel: string) {
    return rulData.value.find((r) => r.channel === channel) || null
  }

  /** Advance to the next carousel channel. */
  function nextCarousel() {
    if (carouselChannels.value.length === 0) return
    carouselIndex.value = (carouselIndex.value + 1) % carouselChannels.value.length
    currentChannel.value = carouselChannels.value[carouselIndex.value]
  }

  /** Manually select a channel (resets the carousel index). */
  function selectChannel(ch: string) {
    currentChannel.value = ch
    const idx = carouselChannels.value.indexOf(ch)
    if (idx >= 0) carouselIndex.value = idx
  }

  // ── Display-name mapping ───────────────────────────────────────────────
  // Internal channelName → display name (e.g. VS-sine → S1).
  // After the device tree is edited, historical alerts re-render with the
  // new name automatically (live mapping, no data migration needed).
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

  /** Internal channelName → display name (returns the input unchanged if not found). */
  function displayName(channel: string | undefined | null): string {
    if (!channel) return '—'
    return displayNameMap.value[channel] || channel
  }

  // ── Auto carousel ──────────────────────────────────────────────────────
  // Defaults to a 15s switch (per the spec); the interval is also readable
  // from theme.carousel.intervalMs.
  let carouselTimer: ReturnType<typeof setInterval> | null = null
  const carouselIntervalMs = ref<number>(15000)

  /** Start the auto carousel (no-op if already running). */
  function startCarousel(intervalMs?: number) {
    if (intervalMs) carouselIntervalMs.value = intervalMs
    if (carouselTimer) return
    carouselTimer = setInterval(() => {
      if (carouselChannels.value.length > 0) {
        nextCarousel()
      }
    }, carouselIntervalMs.value)
  }

  /** Stop the auto carousel. */
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
