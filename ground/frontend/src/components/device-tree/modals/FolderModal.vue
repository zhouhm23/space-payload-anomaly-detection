<script setup lang="ts">
import { ref } from 'vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'

const emit = defineEmits<{ close: [] }>()
const tree = useDeviceTreeStore()
const name = ref('')

function confirm() {
  tree.addFolder(name.value)
  emit('close')
}
</script>

<template>
  <div class="modal-overlay open" @click.self="emit('close')">
    <div class="modal">
      <h3>新建文件夹</h3>
      <label>名称</label>
      <input v-model="name" placeholder="如：仪器柜A" />
      <div class="btn-row">
        <button @click="emit('close')">取消</button>
        <button style="color: var(--accent-blue)" @click="confirm">确认</button>
      </div>
    </div>
  </div>
</template>
