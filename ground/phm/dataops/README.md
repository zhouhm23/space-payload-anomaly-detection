# 数据操作层 (Data Operations Layer)

遥测数据预处理与特征提取。

## 已实现

| 模块 | 职责 |
|------|------|
| `preprocessor.py` | 复用 space 端 `SpacePreprocessor`（缺失值插补 + StandardScaler 标准化） |
| `feature_extractor.py` | `BaseFeatureExtractor` 插件接口（预留） |

## 设计说明

预处理逻辑**复用 space 端实现**，保证天地一致——同一份 `SpacePreprocessor` 代码在轨（检测前）与地基（联合检测前）跑相同流程。

`feature_extractor.py` 定义了统一的特征提取插件契约，本轮不提供具体实现（避免空壳），后续可接入小波/频域/统计特征。
