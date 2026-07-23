<script setup lang="ts">
/**
 * Recursive device-tree node component (folder or sensor).
 *
 * The component self-references to achieve recursion (Vue resolves this
 * via the component's registered name).  Setting name="DeviceTreeNode"
 * lets the template find <DeviceTreeNode> within itself.
 * Collapse state is managed internally per node (default: all expanded).
 *
 * Special sensors (isSpecial) display proprietary attributes:
 * RUL (Remaining Useful Life) degradation time.
 */
import { ref, computed } from 'vue'
import type { DeviceNode } from '@/stores/system'
import { useSystemStore } from '@/stores/system'

defineOptions({ name: 'DeviceTreeNode' })

const props = defineProps<{
  node: DeviceNode
  depth: number
  currentChannel: string
  healthChannels: Record<string, number>
  // folders structure compatibility: {id: {health: 0.95}} or {id: {min: 0.9, mean: 0.95}}
  healthFolders: Record<string, any>
  aggregationStrategy: 'min' | 'mean'
  readonly?: boolean
}>()

const emit = defineEmits<{
  (e: 'select', node: DeviceNode): void
}>()

const store = useSystemStore()

// collapse state (default expanded)
const collapsed = ref(false)

function onClick() {
  if (props.node.type === 'folder') {
    collapsed.value = !collapsed.value
  } else if (props.node.type === 'sensor' && !props.readonly) {
    if (!props.node.isSpecial && props.node.channelName) {
      emit('select', props.node)
    }
  }
}

function healthColor(value: number | undefined): string {
  if (value === undefined || value === null) return '#7a85a8'
  const pct = value * 100
  if (pct >= 80) return '#67c23a'
  if (pct >= 60) return '#e6a23c'
  return '#f56c6c'
}

function channelHealth(name?: string): number | undefined {
  if (!name) return undefined
  const v = props.healthChannels?.[name]
  return typeof v === 'number' ? v : undefined
}

function folderHealth(node: DeviceNode): number | undefined {
  // backend /api/v2/device-tree/ returns folders structure {id: {name, health, strategy, channels}};
  // health is a 0~1 scalar (already aggregated by strategy=min/mean)
  const f = props.healthFolders?.[node.id] || props.healthFolders?.[node.name]
  if (!f) return undefined
  // compatible with two structures: {health: 0.95} or {min: 0.9, mean: 0.95}
  if (typeof f.health === 'number') return f.health
  if (typeof f.min === 'number') return props.aggregationStrategy === 'mean' ? f.mean : f.min
  return undefined
}

function healthPct(v: number | undefined): string {
  if (v === undefined || v === null) return '—'
  return `${Math.round(v * 100)}%`
}

// RUL degradation time (special sensors)
const rulInfo = computed(() => {
  if (props.node.type !== 'sensor' || !props.node.isSpecial) return null
  if (!props.node.channelName) return null
  return store.getRul(props.node.channelName)
})

// RUL progress (0~1, used for colour judgement)
function rulColor(rul: number, max: number): string {
  if (!max) return '#7a85a8'
  const ratio = rul / max
  if (ratio > 0.5) return '#67c23a'
  if (ratio > 0.2) return '#e6a23c'
  return '#f56c6c'
}

const hasChildren = (children?: DeviceNode[]) => Array.isArray(children) && children.length > 0
</script>

<template>
  <div class="tree-node-wrapper">
    <div
      class="tree-node"
      :class="{
        'is-folder': node.type === 'folder',
        'is-sensor': node.type === 'sensor',
        'is-special': node.isSpecial,
        'is-current': node.type === 'sensor' && node.channelName === currentChannel,
      }"
      :style="{ paddingLeft: `${12 + depth * 16}px` }"
      :title="node.description || node.name"
      @click="onClick"
    >
      <!-- Expand/collapse icon -->
      <span v-if="node.type === 'folder' && hasChildren(node.children)" class="expand-icon">
        {{ collapsed ? '▶' : '▼' }}
      </span>
      <span v-else class="expand-icon placeholder"></span>

      <!-- Node icon -->
      <span class="node-icon">
        <template v-if="node.type === 'folder'">📁</template>
        <template v-else-if="node.isSpecial">⚙️</template>
        <template v-else>📊</template>
      </span>

      <!-- Node name (special sensors suffixed with *) -->
      <span class="node-name">
        {{ node.name }}
        <span v-if="node.isSpecial" class="special-mark" title="特殊传感器，不参与轮播">*</span>
      </span>

      <!-- Health badge / RUL degradation time (special sensors) -->
      <span
        v-if="node.type === 'sensor' && node.isSpecial && rulInfo"
        class="health-badge rul-badge"
        :style="{ color: rulColor(rulInfo.rul, rulInfo.max_rul) }"
        :title="`剩余寿命 ${rulInfo.rul.toFixed(1)} ${rulInfo.unit}（上限 ${rulInfo.max_rul}）\n模型: ${rulInfo.model}`"
      >
        ⏳{{ rulInfo.rul.toFixed(0) }} {{ rulInfo.unit }}
      </span>
      <span
        v-else-if="node.type === 'sensor' && node.isSpecial"
        class="health-badge"
        title="RUL 数据加载中"
      >
        ⏳—
      </span>
      <span
        v-else-if="node.type === 'sensor'"
        class="health-badge"
        :style="{ color: healthColor(channelHealth(node.channelName)) }"
        :title="`健康度 = 1 - 异常点数/总点数`"
      >
        {{ healthPct(channelHealth(node.channelName)) }}
      </span>
      <span
        v-else-if="node.type === 'folder'"
        class="health-badge"
        :style="{ color: healthColor(folderHealth(node)) }"
        :title="`文件夹健康度（${aggregationStrategy === 'min' ? '取子通道最小值' : '取子通道平均值'}）`"
      >
        {{ healthPct(folderHealth(node)) }}
      </span>
    </div>

    <!-- Recursive child nodes -->
    <div v-if="node.type === 'folder' && !collapsed && hasChildren(node.children)">
      <DeviceTreeNode
        v-for="child in node.children"
        :key="child.id"
        :node="child"
        :depth="depth + 1"
        :current-channel="currentChannel"
        :health-channels="healthChannels"
        :health-folders="healthFolders"
        :aggregation-strategy="aggregationStrategy"
        :readonly="readonly"
        @select="(n: DeviceNode) => emit('select', n)"
      />
    </div>
  </div>
</template>

<style scoped>
.tree-node-wrapper {
  /* Default: no selection (avoid selecting icons etc. when clicking expand/collapse) */
  user-select: none;
}

.tree-node {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px 6px 0;
  cursor: pointer;
  font-size: 13px;
  color: #e0e6ed;
  transition: background 0.15s;
  white-space: nowrap;
  overflow: hidden;
  border-left: 2px solid transparent;
}

.tree-node:hover {
  background: rgba(64, 158, 255, 0.08);
}

.tree-node.is-current {
  background: rgba(64, 158, 255, 0.15);
  border-left-color: #409eff;
}

.expand-icon {
  width: 12px;
  font-size: 10px;
  color: #7a85a8;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.expand-icon.placeholder {
  width: 12px;
}

.node-icon {
  font-size: 13px;
  flex-shrink: 0;
}

.node-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Name area allows text selection (user can copy sensor name) */
  user-select: text;
  -webkit-user-select: text;
}

.special-mark {
  color: #e6a23c;
  font-weight: bold;
  margin-left: 2px;
}

.health-badge {
  font-size: 11px;
  font-weight: 500;
  font-family: 'Consolas', monospace;
  min-width: 36px;
  text-align: right;
  flex-shrink: 0;
}
</style>
