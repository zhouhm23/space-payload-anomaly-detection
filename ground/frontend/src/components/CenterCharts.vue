<script setup lang="ts">
/**
 * 中图表区（需求书 §中图表区）。
 *
 * 包装 TelemetryCanvas（原生 Canvas，照搬主分支 drawChart）。
 * 三区纵向布局：遥测区:异常分数区:全通道告警点图区 = 4:1:2
 * 顶部显示当前轮播通道指示（含序号 X/Y 和通道名，不遮挡图表）。
 */
import { computed } from 'vue'
import { useSystemStore } from '@/stores/system'
import TelemetryCanvas from './TelemetryCanvas.vue'

const store = useSystemStore()

const currentNum = computed(() => store.carouselIndex + 1)
const total = computed(() => store.carouselChannels.length)
const currentName = computed(() => store.displayName(store.currentChannel) || '未选中')
</script>

<template>
  <div class="center-charts">
    <!-- 顶部：当前轮播通道指示（不遮挡图表） -->
    <div class="chart-header">
      <div class="carousel-indicator">
        <span class="indicator-label">当前通道</span>
        <span class="indicator-value">{{ currentNum }} / {{ total }}</span>
        <span class="indicator-sep">|</span>
        <span class="indicator-name">{{ currentName }}</span>
      </div>
    </div>

    <!-- Canvas 图表（三区 4:1:2） -->
    <div class="chart-area">
      <TelemetryCanvas />
    </div>
  </div>
</template>

<style scoped>
.center-charts {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #0f1530;
}

.chart-header {
  height: 32px;
  display: flex;
  align-items: center;
  padding: 0 14px;
  border-bottom: 1px solid #2a3050;
  flex-shrink: 0;
  background: #1a1f3a;
}

.carousel-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.indicator-label { color: #7a85a8; }
.indicator-value { color: #409eff; font-family: 'Consolas', monospace; font-weight: 500; }
.indicator-sep { color: #2a3050; }
.indicator-name { color: #e0e6ed; font-weight: 500; }

.chart-area {
  flex: 1;
  min-height: 0;
  position: relative;
}
</style>
