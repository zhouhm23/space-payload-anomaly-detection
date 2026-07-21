<script setup lang="ts">
/**
 * 左设备树面板（需求书 §左设备树区）。
 *
 * - 文件夹和传感器都显示健康值
 * - 特殊传感器（非一维数据源）名称后加 `*` 标注，不参与轮播
 * - 通道切换完全由自动轮播控制，左栏不响应点击（仅展示）
 *   （文件夹可点击展开/折叠，这是合理的 UI 交互）
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
