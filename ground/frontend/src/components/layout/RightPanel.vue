<script setup lang="ts">
import AlertList from '@/components/info/AlertList.vue'
import WarningList from '@/components/info/WarningList.vue'

// Health-assessment radar and fault-diagnosis cards are intentionally left
// as placeholders — topology / root-cause analysis are excluded this iteration.
</script>

<template>
  <div class="right-panel">
    <div class="info-card placeholder-card">
      <h4>🚧 健康评估 — 开发中</h4>
      <p>雷达图将在后续版本中实现</p>
    </div>
    <div class="info-card placeholder-card">
      <h4>🚧 故障诊断 — 开发中</h4>
      <p>诊断结果将在后续版本中实现</p>
    </div>
    <WarningList />
    <AlertList />
    <div class="info-card placeholder-card">
      <h4>🚧 RUL 剩余寿命 — 开发中</h4>
      <p>寿命预测将在后续版本中实现</p>
    </div>
  </div>
</template>
