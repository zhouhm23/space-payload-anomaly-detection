# 算法库层 (Algorithm Layer)

异常检测与趋势预测算法封装，定义统一插件基类。

## 已实现

| 模块 | 算法 | 角色 |
|------|------|------|
| `base.py` | `BaseDetector` / `BaseForecaster` ABC | 统一插件基类 |
| `tspulse.py` | TSPulse (IBM Granite, 1M 参数) | 异常检测 |
| `ttm.py` | TTM-R3 (IBM Research, 5M 参数) | 趋势预测 |

## 天地分工

- **天基（space 端）**：`space/anomaly_detection.py` 独立副本，在轨轻量检测，分数随遥测下传
- **地基（ground 端）**：`phm/algorithm/tspulse.py`，用于联合预警流程（实测+预测拼接后整体打分，取预测段）

两端的 TSPulse 代码同源（迁移自原 `ground/anomaly_detection.py`），但实例独立，互不干扰。

## 扩展模型

新增模型只需：
1. 继承 `BaseDetector` 或 `BaseForecaster`
2. 实现对应方法
3. 在 `__init__.py` 注册导出

服务层通过基类引用，无需感知具体实现。
