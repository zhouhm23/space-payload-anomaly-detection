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
import type { DeviceNode, DeviceTreeConfig, SensorPosition } from '@/api/types'

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
  /** Currently-selected folder (new sensors get created inside it). null = top-level. */
  const selectedFolderId = ref<string | null>(null)

  // ---- auto-save (debounced 800ms, silent) ----
  let _autoSaveTimer: ReturnType<typeof setTimeout> | null = null
  function autoSaveConfig(): void {
    if (_autoSaveTimer) clearTimeout(_autoSaveTimer)
    _autoSaveTimer = setTimeout(async () => {
      _autoSaveTimer = null
      try {
        await api.saveConfig({ device_tree: tree.value })
      } catch (e) {
        console.warn('自动保存配置失败:', e)
      }
    }, 800)
  }

  // ---- lookup helpers ----
  /** Fully recursive id lookup (supports arbitrarily nested folders). */
  function findById(id: string): DeviceNode | null {
    function search(nodes: DeviceNode[]): DeviceNode | null {
      for (const n of nodes) {
        if (n.id === id) return n
        if (n.children) {
          const found = search(n.children)
          if (found) return found
        }
      }
      return null
    }
    return search(tree.value)
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

  /** Recursively check whether any sensor exists anywhere in the tree. */
  function anySensorInTree(): boolean {
    function walk(nodes: DeviceNode[]): boolean {
      for (const n of nodes) {
        if (n.sourceId) return true
        if (n.children && walk(n.children)) return true
      }
      return false
    }
    return walk(tree.value)
  }

  /** First sensor's sourceId (depth-first); null if the tree has none. */
  function firstSensorSourceId(): string | null {
    function walk(nodes: DeviceNode[]): string | null {
      for (const n of nodes) {
        if (n.sourceId) return n.sourceId
        if (n.children) {
          const f = walk(n.children)
          if (f) return f
        }
      }
      return null
    }
    return walk(tree.value)
  }

  // ---- mutations ----
  function addSensor(name: string, desc: string, src: string, bs: number, position?: SensorPosition): DeviceNode {
    const node: DeviceNode = {
      id: genId(),
      name: name || '新传感器',
      description: desc,
      type: 'sensor',
      sourceId: src,
      blockSize: bs,
      channelName: sourceToChannel(src) ?? undefined,
      position,
    }
    // Create inside the selected folder if one is active, else top-level
    if (selectedFolderId.value) {
      const folder = findById(selectedFolderId.value)
      if (folder && (folder.type === 'folder' || folder.children)) {
        if (!folder.children) folder.children = []
        folder.children.push(node)
        autoSaveConfig()
        return node
      }
    }
    tree.value.push(node)
    autoSaveConfig()
    return node
  }

  function addFolder(name: string): void {
    const folder: DeviceNode = { id: genId(), name: name || '新文件夹', type: 'folder', children: [] }
    if (selectedFolderId.value) {
      const parent = findById(selectedFolderId.value)
      if (parent && (parent.type === 'folder' || parent.children)) {
        if (!parent.children) parent.children = []
        parent.children.push(folder)
        autoSaveConfig()
        return
      }
    }
    tree.value.push(folder)
    autoSaveConfig()
  }

  /** Recursively collect all distinct ``position.module`` values (for the
   *  device-diagram layout + position-edit dropdown candidates). */
  function allModules(): string[] {
    const seen = new Set<string>()
    function walk(nodes: DeviceNode[]) {
      for (const n of nodes) {
        if (n.position?.module) seen.add(n.position.module)
        if (n.children) walk(n.children)
      }
    }
    walk(tree.value)
    return Array.from(seen)
  }

  /** Merge ``pos`` into the sensor node's existing position (partial update). */
  function setPosition(sensorId: string, pos: Partial<SensorPosition>): void {
    const node = findById(sensorId)
    if (!node) return
    node.position = { ...node.position, ...pos }
    autoSaveConfig()
  }

  function deleteById(id: string): void {
    function removeFrom(nodes: DeviceNode[]): boolean {
      const idx = nodes.findIndex((n) => n.id === id)
      if (idx >= 0) {
        nodes.splice(idx, 1)
        return true
      }
      return nodes.some((n) => n.children && removeFrom(n.children))
    }
    removeFrom(tree.value)
    if (selectedFolderId.value === id) selectedFolderId.value = null
    autoSaveConfig()
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
    // Guard: don't drop a folder into its own descendant (would create a cycle)
    if (dragged.type === 'folder' || dragged.children) {
      const isDescendant = (parent: DeviceNode, id: string): boolean => {
        if (!parent.children) return false
        for (const c of parent.children) {
          if (c.id === id) return true
          if ((c.type === 'folder' || c.children) && isDescendant(c, id)) return true
        }
        return false
      }
      if (isDescendant(dragged, target.id)) return
    }
    deleteById(dragged.id)
    if (target.children || target.type === 'folder') {
      if (!target.children) target.children = []
      target.children.push(dragged)
    } else {
      insertAtParent(dragged, target.id)
    }
    autoSaveConfig()
  }

  // ---- persistence ----
  async function fetchConfig(): Promise<void> {
    try {
      const cfg = await api.getConfig()
      if (cfg.device_tree && cfg.device_tree.length > 0) {
        tree.value = cfg.device_tree
      }
      // Backfill channelName / blockSize for legacy (pre-position) nodes —
      // recursive so nested sensors are covered too.
      function backfill(nodes: DeviceNode[]) {
        for (const n of nodes) {
          if (n.sourceId && !n.channelName) n.channelName = sourceToChannel(n.sourceId) ?? undefined
          if (n.sourceId && !n.blockSize) n.blockSize = 512
          if (n.children) backfill(n.children)
        }
      }
      backfill(tree.value)
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
    selectedFolderId,
    findById,
    selectedSourceId,
    selectedChannelName,
    selectedBlockSize,
    anySensorInTree,
    firstSensorSourceId,
    addSensor,
    addFolder,
    allModules,
    setPosition,
    deleteById,
    dropOn,
    fetchConfig,
    saveConfig,
  }
})
