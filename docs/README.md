# 文档索引

本目录按主题分类存放文本与介绍类材料。根目录只保留 [README.md](../README.md) 作为项目入口。

| 分类 | 路径 | 内容 | 是否入库 |
|------|------|------|----------|
| 架构 | [architecture/](architecture/) | **现行系统设计（架构手册）**、架构总览、完整流程图 | 是 |
| 设计 | [design/](design/) | 设计决策、消融口径、路线图（`PLAN.md`） | 是 |
| 指南 | [guides/](guides/) | 教学向全链路拆解（`how-it-works.md`）、评估口径参考（`metrics.md`） | 是 |
| 执行 | [ops/](ops/) | 夜间批量规格、执行日志、脚本 | 否（本地） |
| 个人 | [personal/](personal/) | 面试准备等个人材料 | 否（本地） |

## 架构

- [architecture/system-design.md](architecture/system-design.md)：**现行系统设计（架构师手册）**——业务流程、模块地图、数据契约、开关/退化、改代码定位表。改代码前先读这份。
- [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)：架构总览、离线/在线数据流示意、延迟口径、组件矩阵
- [architecture/diagrams/doc-rag-full-flow.drawio](architecture/diagrams/doc-rag-full-flow.drawio)：完整流程图（可编辑 draw.io）
- [architecture/diagrams/doc-rag-full-flow.png](architecture/diagrams/doc-rag-full-flow.png)：流程图预览

## 设计

- [design/PLAN.md](design/PLAN.md)：唯一事实源（证据库）。设计取舍、消融、被推翻的结论、复现命令。**不是操作手册**——系统「现在怎么跑」看 architecture/system-design.md
- [design/AGENTIC_RAG.md](design/AGENTIC_RAG.md)：升级为 agentic RAG 的方案 + 自评记录。**P0 已拍板（2026-09-20）**：三条分叉的结论、修正后的分期与验收门槛在 `PLAN.md` §5.5（权威）；本文与 §5.5 冲突处以 §5.5 为准（§10 列了本文被代码复核推翻的 6 处）。代码尚未开工
- [design/adr/](design/adr/)：架构决策记录（ADR）。每条一个决策：背景、选项、拍板、后果。现行：[ADR-0001 升级范围·还债优先](design/adr/ADR-0001-升级范围拍板-还债优先.md) · [ADR-0002 聚合题两段式合成](design/adr/ADR-0002-聚合题两段式合成.md) · [ADR-0003 飞书直连条件转正](design/adr/ADR-0003-飞书直连条件转正.md)
- [design/UPGRADE_2026.md](design/UPGRADE_2026.md)：2026 升级的**执行级实施方案**（Phase 0–3：诊断→冻结库窗口跑两段式验收→库变更与弱项修复→基线重发），含自审修订记录、file:line 级改动点、基线重发对照表。决策依据在 ADR，本文管执行

## 指南

- [guides/how-it-works.md](guides/how-it-works.md)：从 PDF 到可追溯答案的教学文档（不是 API 参考）
- [guides/glossary.md](guides/glossary.md)：**索引式词汇表**——每条一句话定义 + 权威出处指针（规则在 metrics.md、数字在 PLAN.md，本文件不展开）
- [guides/metrics.md](guides/metrics.md)：**评估口径参考**——每个指标的定义、分母与缺失值处理、
  三条噪声地板、判读规则（配对/bootstrap/符号检验/Holm）、七种常见误读。
  **只写规则不写结果值**；数字与放行判定仍归 `design/PLAN.md`

## 执行（本地）

- [ops/night-spec.md](ops/night-spec.md)：夜间批量执行规格
- [ops/night-log.md](ops/night-log.md)：夜间任务日志与对账
- [ops/night-batch.sh](ops/night-batch.sh)：夜间批量脚本（日志写入 `ops/night-log.md`）

## 个人（本地）

- [personal/interview-prep.md](personal/interview-prep.md)：面试叙事重组；数据口径以 `design/PLAN.md` 为准
