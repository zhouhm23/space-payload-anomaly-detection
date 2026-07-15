/* chart.js — placeholder.
 *
 * Per task-9 plan, chart-drawing code (drawChart / drawSubChart /
 * resizeCanvas / rafLoop / roundRect / drag logic / gap collapse /
 * tooltip / crosshair / timeline drag) was meant to live here. However
 * the original dashboard.html keeps ALL JS in a single <script> block
 * with tightly coupled functions sharing closures and global scope
 * (canvas/ctx references, requestAnimationFrame loop, mouse-drag state,
 * dirty-flag closures). Splitting risks breaking function hoisting,
 * event-handler references and the rAF loop.
 *
 * Pragmatic adaptation: the chart code is consolidated VERBATIM into
 * monitor.js (single file) for reliability. monitor.html loads only
 * monitor.js; this file is intentionally empty and kept for parity with
 * the plan's file inventory. Do NOT add code here without first
 * extracting shared globals (window.state / window.C) from monitor.js.
 */
