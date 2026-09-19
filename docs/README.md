# 文档索引

本目录按主题分类存放文本与介绍类材料。根目录只保留 [README.md](../README.md) 作为项目入口。

| 分类 | 路径 | 内容 | 是否入库 |
|------|------|------|----------|
| 架构 | [architecture/](architecture/) | **现行系统设计（架构手册）**、架构总览、完整流程图 | 是 |
| 设计 | [design/](design/) | 设计决策、消融口径、路线图（`PLAN.md`） | 是 |
| 指南 | [guides/](guides/) | 教学向全链路拆解（`how-it-works.md`） | 是 |
| 执行 | [ops/](ops/) | 夜间批量规格、执行日志、脚本 | 否（本地） |
| 个人 | [personal/](personal/) | 面试准备等个人材料 | 否（本地） |

## 架构

- [architecture/system-design.md](architecture/system-design.md)：**现行系统设计（架构师手册）**——业务流程、模块地图、数据契约、开关/退化、改代码定位表。改代码前先读这份。
- [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)：架构总览、离线/在线数据流示意、延迟口径、组件矩阵
- [architecture/diagrams/doc-rag-full-flow.drawio](architecture/diagrams/doc-rag-full-flow.drawio)：完整流程图（可编辑 draw.io）
- [architecture/diagrams/doc-rag-full-flow.png](architecture/diagrams/doc-rag-full-flow.png)：流程图预览

## 设计

- [design/PLAN.md](design/PLAN.md)：唯一事实源（证据库）。设计取舍、消融、被推翻的结论、复现命令。**不是操作手册**——系统「现在怎么跑」看 architecture/system-design.md
- [design/AGENTIC_RAG.md](design/AGENTIC_RAG.md)：升级为 agentic RAG 的方案（草案）。§0 三条分叉待拍板，未开工

## 指南

- [guides/how-it-works.md](guides/how-it-works.md)：从 PDF 到可追溯答案的教学文档（不是 API 参考）

## 执行（本地）

- [ops/night-spec.md](ops/night-spec.md)：夜间批量执行规格
- [ops/night-log.md](ops/night-log.md)：夜间任务日志与对账
- [ops/night-batch.sh](ops/night-batch.sh)：夜间批量脚本（日志写入 `ops/night-log.md`）

## 个人（本地）

- [personal/interview-prep.md](personal/interview-prep.md)：面试叙事重组；数据口径以 `design/PLAN.md` 为准
