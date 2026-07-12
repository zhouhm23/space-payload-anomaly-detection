<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useDeviceTreeStore, sourceToChannel } from '@/stores/deviceTree'
import { usePoll } from '@/composables/usePoll'
import type { SensorPosition } from '@/api/types'

const props = defineProps<{ nodeId: string }>()
const emit = defineEmits<{ close: [] }>()

const tree = useDeviceTreeStore()
const { fetchBlock } = usePoll()

const name = ref('')
const desc = ref('')
const src = ref('file:NASA-MSL/C-1')
const bs = ref(512)

// ---- position editing (Slice 0) ----
const hasPosition = ref(false)
const posModule = ref('')
const posX = ref(0.5)
const posY = ref(0.5)

/** True when editing a folder (only name is editable). */
const isFolder = ref(false)

/** Module dropdown candidates: distinct modules already used in the tree. */
const moduleOptions = computed(() => tree.allModules())

/** 标准数据源列表（与天基 DAQ_CONFIG 一致）。若节点 sourceId 不在其中，下拉补一个临时选项。 */
const STANDARD_SOURCES = ['file:NASA-MSL/C-1', 'file:NASA-MSL/D-14', 'virtual:sine', 'virtual:multi_sine']
const srcNotInList = computed(() => src.value !== '' && !STANDARD_SOURCES.includes(src.value))

watch(
  () => props.nodeId,
  (id) => {
    const node = tree.findById(id)
    if (!node) return
    name.value = node.name || ''
    isFolder.value = !!(node.type === 'folder' || node.children)
    if (isFolder.value) return // folders: name-only edit
    desc.value = node.description || ''
    src.value = node.sourceId || 'file:NASA-MSL/C-1'
    bs.value = node.blockSize || 512
    // Load existing position (if any)
    const p = node.position
    if (p && (p.x !== undefined || p.y !== undefined)) {
      hasPosition.value = true
      posModule.value = p.module || ''
      posX.value = p.x ?? 0.5
      posY.value = p.y ?? 0.5
    } else {
      hasPosition.value = false
      posModule.value = ''
      posX.value = 0.5
      posY.value = 0.5
    }
  },
  { immediate: true },
)

function confirm() {
  const node = tree.findById(props.nodeId)
  if (!node) return
  node.name = name.value || node.name
  if (isFolder.value) {
    // Folder: only rename, then done
    emit('close')
    return
  }
  node.description = desc.value
  node.sourceId = src.value
  node.channelName = sourceToChannel(src.value) ?? undefined
  node.blockSize = bs.value

  if (hasPosition.value) {
    const position: SensorPosition = { x: posX.value, y: posY.value }
    if (posModule.value.trim()) position.module = posModule.value.trim()
    node.position = position
  } else {
    delete node.position
  }

  emit('close')
  if (tree.selectedId === node.id) fetchBlock(bs.value)
}
</script>

<template>
  <div class="modal-overlay open" @click.self="emit('close')">
    <div class="modal">
      <h3>{{ isFolder ? '重命名文件夹' : '编辑传感器配置' }}</h3>
      <label>名称</label>
      <input v-model="name" :placeholder="isFolder ? '文件夹名称' : '传感器名称'" />
      <template v-if="!isFolder">
      <label>描述</label>
      <input v-model="desc" placeholder="传感器描述" />
      <label>数据源（天基采集通道）</label>
      <select v-model="src">
        <option v-if="srcNotInList" :value="src">{{ src }}（当前值，不在标准列表）</option>
        <option value="file:NASA-MSL/C-1">NASA-MSL C-1（温度）</option>
        <option value="file:NASA-MSL/D-14">NASA-MSL D-14</option>
        <option value="virtual:sine">虚拟正弦波</option>
        <option value="virtual:multi_sine">虚拟多谐波</option>
      </select>
      <label>区块大小</label>
      <input v-model.number="bs" type="number" min="64" max="65536" step="64" />

      <div class="position-section">
        <label class="checkbox-row">
          <input v-model="hasPosition" type="checkbox" />
          <span>设置示意图位置</span>
        </label>
        <div v-if="hasPosition" class="position-fields">
          <label>所属模块（可选，留空归入"未分组"）</label>
          <input
            v-model="posModule"
            list="module-candidates-edit"
            placeholder="如：电源模块（可输入新值）"
          />
          <datalist id="module-candidates-edit">
            <option v-for="m in moduleOptions" :key="m" :value="m" />
          </datalist>
          <label>X 位置：{{ posX.toFixed(2) }}</label>
          <input v-model.number="posX" type="range" min="0" max="1" step="0.01" />
          <label>Y 位置：{{ posY.toFixed(2) }}</label>
          <input v-model.number="posY" type="range" min="0" max="1" step="0.01" />
          <div class="mini-preview">
            <div
              class="mini-preview-dot"
              :style="{ left: posX * 100 + '%', top: posY * 100 + '%' }"
            />
          </div>
        </div>
      </div>
      </template>

      <div class="btn-row">
        <button @click="emit('close')">取消</button>
        <button style="color: var(--accent-blue)" @click="confirm">确认</button>
      </div>
    </div>
  </div>
</template>
