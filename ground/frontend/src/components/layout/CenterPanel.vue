<script setup lang="ts">
import SensorGrid from '@/components/dashboard/SensorGrid.vue'
import DeviceDiagram from '@/components/device-diagram/DeviceDiagram.vue'
</script>

<template>
  <div class="center-panel">
    <DeviceDiagram style="flex: 1" />
    <SensorGrid style="flex: 2" />
  </div>
</template>
