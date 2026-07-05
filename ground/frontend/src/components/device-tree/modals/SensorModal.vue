<script setup lang="ts">
import { ref, watch } from 'vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { usePoll } from '@/composables/usePoll'

const props = defineProps<{ modelValue?: boolean }>()
const emit = defineEmits<{ close: [] }>()

const tree = useDeviceTreeStore()
const { fetchBlock } = usePoll()

const name = ref('')
const desc = ref('')
const src = ref('file:NASA-MSL/C-1')
const bs = ref(512)

function confirm() {
  const node = tree.addSensor(name.value, desc.value, src.value, bs.value)
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
      <div class="btn-row">
        <button @click="emit('close')">取消</button>
        <button style="color: var(--accent-blue)" @click="confirm">确认</button>
      </div>
    </div>
  </div>
</template>
