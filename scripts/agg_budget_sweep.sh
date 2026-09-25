#!/usr/bin/env bash
# A4 聚合题上下文预算扫线（**下一轮跑**；本轮只交付「命令存在 + 参数正确」，见 PLAN §5.7）。
#
# 背景（私有语料 A6/C25 实测，PLAN §5.5/§5.6）：keypoint_recall 0.1951（6/6）→ 0.4623（25/25），
# 代价输入 3.4×、聚合 p95 破 8s SLO。新底座上哪一档是「p95 ≤5s 约束下的最优」由本轮扫线回答。
#
# ⚠️⇧ 方法学坑（必须先读，benchmark-rgb.md:202-205 同款教训）：
#   只动 --max-contexts 是**半个杠杆**——进 LLM 的块数 = min(max_contexts, rerank.top_n)，
#   清单只有 8 格时 --max-contexts 25 会被截在 8（E2 臂 B 已经证伪过一次，PLAN §5.5）。
#   所以每一臂必须**同时**给 --top-n N --max-contexts N，两臂才真正相差预算。
#
# 用法（下一轮，付费 ≈≤¥5；7 臂一次跑齐进同一 Holm 家族，禁止分多次跑）：
#   bash scripts/agg_budget_sweep.sh                 # 全量 7 臂（真实合成计费）
#   bash scripts/agg_budget_sweep.sh --smoke         # 冒烟：--limit 1 --retrieval-only，¥0
#
# 拍板规则（先于看数写定，PLAN §5.7；2026-09-26 修订为双曲线同向）：
#   1) 每档取「聚合端到端 p95 ≤5s 约束下的 keypoint_recall」；
#   2) 全 31 条 argmax 与 K≥3 子集（n=15）argmax **一致**才算定档；不一致 = 分辨率
#      不足、不定档（16/31 条 K=1，单条翻转 ≈3.2pt，与噪声地板同量级）；
#   3) 地板先行：B5 三条噪声地板（--repeat --fresh-answers）先测，argmax 才有
#      噪声带可对照；
#   4) 若最优点仍是 6，维持 6 并把曲线留档——这同样是合格结论。
# 路由前置已闭（2026-09-26 探针）：新底座聚合路由率 30/31=0.968，七臂量的确实是聚合路。

set -euo pipefail

ARMS=(6 8 10 12 15 20 25)
OUT_DIR="${OUT_DIR:-data/eval}"
GOLD="${GOLD:-data/eval/gold_core_agg.json}"
KB="${KB:-doc_rag_sample}"

if [[ "${1:-}" == "--smoke" ]]; then
  # 冒烟：单条 + 只检索（不调合成）→ ¥0，验证命令与参数形状
  EXTRA=(--limit 1 --retrieval-only)
else
  # 正式：关合成缓存（延迟是真数，keypoint 判据才可比）
  EXTRA=(--fresh-answers)
fi

for N in "${ARMS[@]}"; do
  echo "=== 臂 N=${N}（--top-n ${N} --max-contexts ${N}）==="
  uv run doc-rag eval \
    --gold "${GOLD}" \
    --kb "${KB}" \
    --rewrite --rerank \
    --top-n "${N}" \
    --max-contexts "${N}" \
    "${EXTRA[@]}" \
    --ragas-out "${OUT_DIR}/agg_sweep_${N}.json"
done

echo "7 臂完成 → ${OUT_DIR}/agg_sweep_{6,8,10,12,15,20,25}.json"
echo "判读入口："
echo "  uv run doc-rag compare-retrieval \\"
for N in "${ARMS[@]}"; do
  echo "    --group \"N${N}=${OUT_DIR}/agg_sweep_${N}.json\" \\"
done
echo "    --metric keypoint_hit_ratio"
