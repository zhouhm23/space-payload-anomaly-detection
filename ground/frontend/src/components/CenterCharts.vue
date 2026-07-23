<script setup lang="ts">
/**
 * Center chart area (spec centre-chart panel).
 *
 * Wraps TelemetryCanvas (native Canvas, ported from the main branch's
 * monitor.js drawChart). Three vertical regions with ratio
 * telemetry:anomaly-score:all-channel-alert-map = 4:1:2.
 * The header shows the current carousel channel indicator (index X/Y and
 * the channel name) without overlapping the chart.
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
    <!-- Header: current carousel channel indicator (does not overlap the chart) -->
    <div class="chart-header">
      <div class="carousel-indicator">
        <span class="indicator-label">当前通道</span>
        <span class="indicator-value">{{ currentNum }} / {{ total }}</span>
        <span class="indicator-sep">|</span>
        <span class="indicator-name">{{ currentName }}</span>
      </div>
    </div>

    <!-- Canvas chart (three regions at 4:1:2) -->
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
