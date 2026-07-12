/**
 * Health / alerts / warnings store — polls the new PHM endpoints and
 * exposes reactive snapshots for the dashboard cards, alert bar and
 * warning bar.
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/api/client'
import type { AlertItem, HealthResponse, SensorItem, WarningItem } from '@/api/types'

export const useHealthStore = defineStore('health', () => {
  const systemHealth = ref(100)
  const channelHealth = ref<Record<string, number>>({})
  const folders = ref<Record<string, { name: string; health: number; strategy: string; channels: string[] }>>({})
  const threshold = ref(0.7)
  const sensors = ref<SensorItem[]>([])
  const alerts = ref<AlertItem[]>([])
  const warnings = ref<WarningItem[]>([])

  async function refreshAll(): Promise<void> {
    await Promise.all([refreshHealth(), refreshAlerts(), refreshWarnings(), refreshSensors()])
  }

  async function refreshHealth(): Promise<void> {
    try {
      const r: HealthResponse = await api.health()
      systemHealth.value = r.system
      channelHealth.value = r.channels
      folders.value = r.folders || {}
      threshold.value = r.threshold
    } catch {
      /* keep last good values */
    }
  }

  async function refreshAlerts(): Promise<void> {
    try {
      const r = await api.alerts()
      alerts.value = r.alerts
    } catch {
      /* ignore */
    }
  }

  async function refreshWarnings(): Promise<void> {
    try {
      const r = await api.warnings()
      warnings.value = r.warnings
    } catch {
      /* ignore */
    }
  }

  async function refreshSensors(): Promise<void> {
    try {
      const r = await api.sensors()
      sensors.value = r.sensors
      systemHealth.value = r.system_health
    } catch {
      /* ignore */
    }
  }

  return {
    systemHealth,
    channelHealth,
    folders,
    threshold,
    sensors,
    alerts,
    warnings,
    refreshAll,
    refreshHealth,
    refreshAlerts,
    refreshWarnings,
    refreshSensors,
  }
})
