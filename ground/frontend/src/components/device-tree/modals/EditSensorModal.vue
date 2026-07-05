<script setup lang="ts">
import { ref, watch } from 'vue'
import { useDeviceTreeStore, sourceToChannel } from '@/stores/deviceTree'
import { usePoll } from '@/composables/usePoll'

const props = defineProps<{ nodeId: string }>()
const emit = defineEmits<{ close: [] }>()

const tree = useDeviceTreeStore()
const { fetchBlock } = usePoll()

const name = ref('')
const desc = ref('')
const src = ref('file:NASA-MSL/C-1')
const bs = ref(512)

watch(
  () => props.nodeId,
  (id) => {
    const node = tree.findById(id)
    if (!node) return
    name.value = node.name || ''
    desc.value = node.description || ''
    src.value = node.sourceId || 'file:NASA-MSL/C-1'
    bs.value = node.blockSize || 512
  },
  { immediate: true },
)

function confirm() {
  const node = tree.findById(props.nodeId)
  if (!node) return
  node.name = name.value || node.name
  node.description = desc.value
  node.sourceId = src.value
  node.channelName = sourceToChannel(src.value) ?? undefined
  node.blockSize = bs.value
  emit('close')
  if (tree.selectedId === node.id) fetchBlock(bs.value)
}
</script>

<template>
  <div class="modal-overlay open" @click.self="emit('close')">
    <div class="modal">
      <h3>编辑传感器配置</h3>
      <label>名称</label>
      <input v-model="name" placeholder="传感器名称" />
      <label>描述</label>
      <input v-model="desc" placeholder="传感器描述" />
      <label>数据源 (sourceId)</label>
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
