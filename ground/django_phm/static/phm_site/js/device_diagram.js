/* device_diagram.js — placeholder.
 *
 * Per task-9 plan, device-diagram code (drawDeviceDiagram /
 * groupSensorsByModule / gridDims / diagramHitTest / healthColorHex /
 * mousemove+click handler) was meant to live here. Same rationale as
 * chart.js: the original keeps ALL JS in one <script> block sharing
 * closures (diagramCanvas/diagramCtx/diagramLayout/dpr), global state
 * and the rAF loop. Splitting risks breaking hoisting/handler refs.
 *
 * The device-diagram code is consolidated VERBATIM into monitor.js.
 * monitor.html loads only monitor.js; this file is intentionally empty
 * and kept for parity with the plan's file inventory.
 */
