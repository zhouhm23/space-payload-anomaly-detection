<script setup lang="ts">
import { onMounted } from 'vue'
import HeaderBar from '@/components/layout/HeaderBar.vue'
import LeftPanel from '@/components/layout/LeftPanel.vue'
import CenterPanel from '@/components/layout/CenterPanel.vue'
import RightPanel from '@/components/layout/RightPanel.vue'
import BottomPanel from '@/components/layout/BottomPanel.vue'
import { useDeviceTreeStore } from '@/stores/deviceTree'
import { useLinkStore } from '@/stores/link'

const tree = useDeviceTreeStore()
const link = useLinkStore()

onMounted(async () => {
  link.startClock()
  await tree.fetchConfig()
  // auto-select first sensor node if none selected
  if (!tree.selectedId) {
    const first = tree.tree.find((d) => d.sourceId) || tree.tree[0]
    if (first) tree.selectedId = first.id
  }
})
</script>

<template>
  <!-- 顶部状态栏 -->
  <HeaderBar />
  <!-- 主体 -->
  <div class="main-content">
    <LeftPanel />
    <CenterPanel />
    <RightPanel />
  </div>
  <!-- 底部数据流区域 -->
  <BottomPanel />
</template>
