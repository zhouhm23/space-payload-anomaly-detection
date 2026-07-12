<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { usePoll } from '@/composables/usePoll'
import type { SensorPosition } from '@/api/types'

const props = defineProps<{ modelValue?: boolean }>()
const emit = defineEmits<{ close: [] }>()

const tree = useDeviceTreeStore()
const { fetchBlock } = usePoll()

const name = ref('')
const desc = ref('')
const src = ref('file:NASA-MSL/C-1')
const bs = ref(512)

// ---- position editing (Slice 0) ----
const hasPosition = ref(false) // user can opt in
const posModule = ref('')
const posX = ref(0.5)
const posY = ref(0.5)

/** Module dropdown candidates: distinct modules already used in the tree. */
const moduleOptions = computed(() => tree.allModules())

function confirm() {
  let position: SensorPosition | undefined
  if (hasPosition.value) {
    position = { x: posX.value, y: posY.value }
    if (posModule.value.trim()) position.module = posModule.value.trim()
  }
  const node = tree.addSensor(name.value, desc.value, src.value, bs.value, position)
  tree.selectedId = node.id
  emit('close')
  fetchBlock(bs.value)
}
</script>

<template>
  <div class="modal-overlay open" @click.self="emit('close')">
    <div class="modal">
      <h3>新建传感器</h3>
      <label>名称</label>
      <input v-model="name" placeholder="如：温度传感器" />
      <label>描述</label>
      <input v-model="desc" placeholder="如：仪器柜A热控" />
      <label>数据源</label>
      <select v-model="src">
        <option value="file:NASA-MSL/C-1">NASA-MSL C-1</option>
        <option value="file:NASA-MSL/D-14">NASA-MSL D-14</option>
        <option value="file:NASA-SMAP/E-1">NASA-SMAP E-1</option>
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
            list="module-candidates"
            placeholder="如：电源模块（可输入新值）"
          />
          <datalist id="module-candidates">
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

      <div class="btn-row">
        <button @click="emit('close')">取消</button>
        <button style="color: var(--accent-blue)" @click="confirm">确认</button>
      </div>
    </div>
  </div>
</template>
