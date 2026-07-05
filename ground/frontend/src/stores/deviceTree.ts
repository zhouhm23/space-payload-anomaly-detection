/**
 * Device tree store — CRUD, drag-and-drop, sourceId→channelName mapping,
 * persistence via /api/config.
 *
 * Ported from the legacy HTML's ``deviceTreeData`` + ``_findById`` /
 * ``_deleteById`` / ``_insertAtParent`` / ``sourceToChannel`` helpers.
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/api/client'
import type { DeviceNode, DeviceTreeConfig } from '@/api/types'

/** Map a source_id to the backend channel name (mirrors sensor_source.py). */
export function sourceToChannel(sourceId?: string): string | null {
  if (!sourceId) return null
  if (sourceId.startsWith('virtual:')) {
    return 'VS-' + sourceId.slice('virtual:'.length)
  }
  if (sourceId.startsWith('file:')) {
    const rest = sourceId.slice('file:'.length)
    const parts = rest.split('/')
    return parts[parts.length - 1]
  }
  return sourceId
}

export function genId(): string {
  return 'n_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6)
}

const DEFAULT_TREE: DeviceNode[] = [
  {
    id: 'thermal',
    name: '热控系统',
    children: [
      { id: 'pump', name: '泵组' },
      { id: 'filter', name: '过滤器' },
      { id: 'cooler', name: '制冷机' },
      { id: 'controller', name: '控制器' },
      { id: 'sensor', name: '传感器' },
    ],
  },
  {
    id: 'power',
    name: '供配电系统',
    children: [{ id: 'battery', name: '电池' }],
  },
]

export const useDeviceTreeStore = defineStore('deviceTree', () => {
  const tree = ref<DeviceNode[]>(structuredClone(DEFAULT_TREE))
  const selectedId = ref<string | null>(null)

  // ---- lookup helpers ----
  function findById(id: string): DeviceNode | null {
    for (const d of tree.value) {
      if (d.id === id) return d
      if (d.children) {
        const f = d.children.find((c) => c.id === id)
        if (f) return f
      }
    }
    return null
  }

  function selectedSourceId(): string | null {
    const node = selectedId.value ? findById(selectedId.value) : null
    if (!node) return null
    if (node.sourceId) return node.sourceId
    if (node.children) {
      const child = node.children.find((c) => c.sourceId)
      if (child) return child.sourceId ?? null
    }
    return null
  }

  function selectedChannelName(): string | null {
    const node = selectedId.value ? findById(selectedId.value) : null
    if (!node) return null
    if (node.channelName) return node.channelName
    if (node.sourceId) return sourceToChannel(node.sourceId)
    if (node.children) {
      const child = node.children.find((c) => c.sourceId || c.channelName)
      if (child) return child.channelName ?? sourceToChannel(child.sourceId)
    }
    return null
  }

  function selectedBlockSize(fallback = 512): number {
    const node = selectedId.value ? findById(selectedId.value) : null
    if (node && node.blockSize) return node.blockSize
    return fallback
  }

  function anySensorInTree(): boolean {
    for (const d of tree.value) {
      if (d.sourceId) return true
      if (d.children) {
        for (const c of d.children) {
          if (c.sourceId) return true
        }
      }
    }
    return false
  }

  function firstSensorSourceId(): string | null {
    for (const d of tree.value) {
      if (d.sourceId) return d.sourceId
      if (d.children) {
        const c = d.children.find((x) => x.sourceId)
        if (c) return c.sourceId ?? null
      }
    }
    return null
  }

  // ---- mutations ----
  function addSensor(name: string, desc: string, src: string, bs: number): DeviceNode {
    const node: DeviceNode = {
      id: genId(),
      name: name || '新传感器',
      description: desc,
      type: 'sensor',
      sourceId: src,
      blockSize: bs,
      channelName: sourceToChannel(src) ?? undefined,
    }
    tree.value.push(node)
    return node
  }

  function addFolder(name: string): void {
    tree.value.push({ id: genId(), name: name || '新文件夹', type: 'folder', children: [] })
  }

  function deleteById(id: string): void {
    for (let i = 0; i < tree.value.length; i++) {
      if (tree.value[i].id === id) {
        tree.value.splice(i, 1)
        return
      }
      if (tree.value[i].children) {
        tree.value[i].children = tree.value[i].children!.filter((c) => c.id !== id)
      }
    }
  }

  /** Move ``moved`` to just before ``beforeId`` (sibling), like legacy. */
  function insertAtParent(moved: DeviceNode, beforeId: string): void {
    for (const d of tree.value) {
      if (d.id === beforeId) {
        const i = tree.value.indexOf(d)
        tree.value.splice(i, 0, moved)
        return
      }
      if (d.children) {
        const idx = d.children.findIndex((c) => c.id === beforeId)
        if (idx >= 0) {
          d.children.splice(idx, 0, moved)
          return
        }
      }
    }
    tree.value.push(moved)
  }

  /** Drop ``dragged`` onto ``target`` — replicate the legacy drop logic. */
  function dropOn(dragged: DeviceNode, target: DeviceNode): void {
    if (dragged.id === target.id) return
    deleteById(dragged.id)
    if (target.children || target.type === 'folder') {
      if (!target.children) target.children = []
      target.children.push(dragged)
    } else {
      insertAtParent(dragged, target.id)
    }
  }

  // ---- persistence ----
  async function fetchConfig(): Promise<void> {
    try {
      const cfg = await api.getConfig()
      if (cfg.device_tree && cfg.device_tree.length > 0) {
        tree.value = cfg.device_tree
      }
      // backfill channelName / blockSize for legacy nodes
      for (const d of tree.value) {
        if (d.sourceId && !d.channelName) d.channelName = sourceToChannel(d.sourceId) ?? undefined
        if (d.sourceId && !d.blockSize) d.blockSize = 512
        if (d.children) {
          for (const c of d.children) {
            if (c.sourceId && !c.channelName) c.channelName = sourceToChannel(c.sourceId) ?? undefined
            if (c.sourceId && !c.blockSize) c.blockSize = 512
          }
        }
      }
    } catch (e) {
      console.warn('加载配置失败，使用默认设备树:', e)
    }
  }

  async function saveConfig(): Promise<void> {
    const body: DeviceTreeConfig = { device_tree: tree.value }
    await api.saveConfig(body)
  }

  return {
    tree,
    selectedId,
    findById,
    selectedSourceId,
    selectedChannelName,
    selectedBlockSize,
    anySensorInTree,
    firstSensorSourceId,
    addSensor,
    addFolder,
    deleteById,
    dropOn,
    fetchConfig,
    saveConfig,
  }
})
