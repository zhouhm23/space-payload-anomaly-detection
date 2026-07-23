<script setup lang="ts">
/**
 * Left device-tree panel (spec left device-tree panel).
 *
 * - Both folders and sensors display a health value so the operator can
 *   spot degraded subsystems at a glance.
 * - Special sensors (non-1D data sources) are suffixed with `*` and are
 *   excluded from the auto-carousel because their data format differs.
 * - Channel switching is fully controlled by the auto-carousel; the left
 *   panel is display-only and does not respond to sensor clicks.
 * - Folders may still be clicked to expand or collapse, which is a
 *   reasonable navigation interaction.
 */
import { computed } from 'vue'
import { useSystemStore } from '@/stores/system'
import DeviceTreeNode from './DeviceTreeNode.vue'

const store = useSystemStore()

const tree = computed(() => store.deviceTree?.device_tree || [])
const healthChannels = computed(() => store.deviceTree?.health?.channels || {})
const healthFolders = computed<Record<string, any>>(() => store.deviceTree?.health?.folders || {})
const strategy = computed(() => store.deviceTree?.aggregation_strategy || 'min')
</script>

<template>
  <div class="device-tree-panel">
    <div class="panel-header">
      <span class="panel-title">设备树</span>
      <span class="panel-subtitle">
        {{ store.carouselChannels.length }} 通道参与轮播
      </span>
    </div>
    <div class="tree-body">
      <div v-if="!tree.length" class="empty-tip">暂无设备树数据</div>

      <DeviceTreeNode
        v-for="node in tree"
        :key="node.id"
        :node="node"
        :depth="0"
        :current-channel="store.currentChannel"
        :health-channels="healthChannels"
        :health-folders="healthFolders"
        :aggregation-strategy="strategy"
        :readonly="true"
      />
    </div>
  </div>
</template>

<style scoped>
.device-tree-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #0f1530;
}

.panel-header {
  padding: 12px 14px;
  border-bottom: 1px solid #2a3050;
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex-shrink: 0;
}

.panel-title {
  font-size: 14px;
  font-weight: 500;
  color: #409eff;
  letter-spacing: 1px;
}

.panel-subtitle {
  font-size: 11px;
  color: #7a85a8;
}

.tree-body {
  flex: 1;
  overflow-y: auto;
  padding: 6px 0;
}

.empty-tip {
  padding: 20px;
  text-align: center;
  color: #7a85a8;
  font-size: 12px;
}
</style>
