/**
 * Link status store — UTC clock + ground-space link indicator.
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'

export type LinkState = 'normal' | 'waiting' | 'loss'

export const useLinkStore = defineStore('link', () => {
  const utcTime = ref('')
  const linkState = ref<LinkState>('waiting')

  function startClock(): void {
    const pad = (n: number) => String(n).padStart(2, '0')
    const update = () => {
      const d = new Date()
      utcTime.value = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
    }
    update()
    setInterval(update, 1000)
  }

  function updateFromPolling(playing: boolean, hasTimer: boolean, hasBlocks: boolean): void {
    if (playing && hasTimer) {
      linkState.value = 'normal'
    } else if (!playing && !hasBlocks) {
      linkState.value = 'waiting'
    }
  }

  return { utcTime, linkState, startClock, updateFromPolling }
})
