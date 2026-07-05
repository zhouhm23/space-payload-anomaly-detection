# 算法库层映射（前端）

前端通过 `/api/forecast` 触发 TTM-R3 预测，通过 `/api/poll` 获取 TSPulse 检测分数。
算法本体在后端 `algorithm/`，前端仅消费结果。

- **后端实体**：`src/ground/phm/algorithm/{tspulse,ttm,base}.py`
- **前端对应**：`src/composables/useForecast.ts`
