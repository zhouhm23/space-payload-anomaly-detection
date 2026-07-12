<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { usePoll } from '@/composables/usePoll'
import { useTelemetryStore } from '@/stores/telemetry'
import TreeNode from './TreeNode.vue'
import SensorModal from './modals/SensorModal.vue'
import FolderModal from './modals/FolderModal.vue'
import EditSensorModal from './modals/EditSensorModal.vue'

const tree = useDeviceTreeStore()
const telemetry = useTelemetryStore()
const { fetchBlock } = usePoll()

const showSensorModal = ref(false)
const showFolderModal = ref(false)
const showEditModal = ref(false)
const editingId = ref<string | null>(null)

// flattened node list with depth info for rendering
const flatNodes = computed(() => {
  const out: { node: any; depth: number }[] = []
  function walk(nodes: any[], depth: number) {
    for (const n of nodes) {
      out.push({ node: n, depth })
      if (n.children && n.children.length > 0) walk(n.children, depth + 1)
    }
  }
  walk(tree.tree, 0)
  return out
})

let draggedNode: any = null

function onDragStart(node: any) {
  draggedNode = node
}
function onDrop(target: any) {
  if (!draggedNode) return
  tree.dropOn(draggedNode, target)
  draggedNode = null
}

async function selectNode(id: string) {
  tree.selectedId = id
  const node = tree.findById(id)
  if (node && (node.type === 'folder' || node.children)) {
    // Folder click toggles "create-inside-me" selection (no channel to fetch)
    tree.selectedFolderId = tree.selectedFolderId === id ? null : id
    return
  }
  // Sensor: fetch latest data for the newly-selected channel immediately
  const bs = tree.selectedBlockSize(512)
  await fetchBlock(bs)
}

function deleteNode(id: string, e: Event) {
  e.stopPropagation()
  tree.deleteById(id)
}

function openEdit(id: string) {
  editingId.value = id
  showEditModal.value = true
}

async function saveTree() {
  try {
    await tree.saveConfig()
    alert('配置已保存')
  } catch (e) {
    alert('保存失败: ' + e)
  }
}
</script>

<template>
  <div class="tree-actions">
    <button @click="showSensorModal = true">➕ 传感器</button>
    <button @click="showFolderModal = true">📁 文件夹</button>
    <button style="margin-left: auto" @click="saveTree">💾 保存</button>
  </div>
  <ul class="device-tree">
    <TreeNode
      v-for="item in flatNodes"
      :key="item.node.id"
      :node="item.node"
      :depth="item.depth"
      :active="tree.selectedId === item.node.id || tree.selectedFolderId === item.node.id"
      @select="selectNode(item.node.id)"
      @delete="deleteNode(item.node.id, $event)"
      @dblclick="openEdit(item.node.id)"
      @dragstart="onDragStart(item.node)"
      @drop="onDrop(item.node)"
    />
  </ul>

  <SensorModal v-if="showSensorModal" @close="showSensorModal = false" />
  <FolderModal v-if="showFolderModal" @close="showFolderModal = false" />
  <EditSensorModal v-if="showEditModal" :node-id="editingId!" @close="showEditModal = false" />
</template>
