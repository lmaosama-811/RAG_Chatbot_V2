"""
eval/phase4_report/generate_report.py
========================================
PHASE 4: Tạo báo cáo markdown từ metrics JSON.

INPUT  : eval/results/metrics_YYYYMMDD.json  (mới nhất tự động)
         eval/data/qa_pairs.json
OUTPUT : eval/results/report_YYYYMMDD.md

Chạy lệnh:
    python3 -m eval.phase4_report.generate_report
"""
import sys, os, json, glob, statistics
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)
EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from eval.config import QA_PAIRS_PATH, RESULTS_DIR


def find_latest_metrics() -> str:
    files = glob.glob(os.path.join(RESULTS_DIR, 'metrics_*.json'))
    if not files:
        raise FileNotFoundError(
            f'Không tìm thấy metrics_*.json trong: {RESULTS_DIR}\n'
            f'Chạy phase3 trước: python3 -m eval.phase3_compute.compute_metrics'
        )
    return max(files, key=os.path.getmtime)


def load_latest_optional(pattern: str):
    """
    Tìm file JSON mới nhất khớp với pattern trong RESULTS_DIR.
    Trả về None nếu không có — report vẫn chạy bình thường khi chưa
    chạy component isolated eval.
    """
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return None
    path = max(files, key=os.path.getmtime)
    print(f'[report] Optional loaded: {os.path.basename(path)}')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def fmt(v, decimals=4) -> str:
    """Format metric value. None → N/A."""
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:.{decimals}f}'
    return str(v)


def score_badge(v, thresholds=((0.8, '🟢'), (0.5, '🟡'), (0.0, '🔴'))) -> str:
    """Trả về emoji màu tương ứng với mức điểm."""
    if v is None:
        return '⚪'
    for threshold, badge in thresholds:
        if v >= threshold:
            return badge
    return '🔴'


def ms_to_s(v) -> str:
    """Chuyển ms → giây, 2 chữ số thập phân."""
    if v is None:
        return 'N/A'
    return f'{v / 1000:.2f}s'


def safe_mean(lst):
    vals = [x for x in lst if x is not None]
    return round(statistics.mean(vals), 4) if vals else None


def run():
    metrics_path = find_latest_metrics()
    print(f'[report] Using: {metrics_path}')

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    with open(QA_PAIRS_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    # Load isolated component metrics (None nếu chưa chạy)
    gen_metrics    = load_latest_optional('generator_metrics_*.json')
    rerank_metrics = load_latest_optional('reranker_metrics_*.json')
    gen_per_map    = {r['test_id']: r for r in gen_metrics['per_case']}    if gen_metrics    else {}
    rerank_per_map = {r['test_id']: r for r in rerank_metrics['per_case']} if rerank_metrics else {}

    agg  = metrics['aggregate']
    per  = metrics['per_test_case']
    cfg  = metrics.get('config', {})
    rid  = metrics.get('run_id', 'unknown')

    # Map test_id → QA metadata
    notes_map    = {qa['id']: qa.get('notes', '')         for qa in qa_pairs}
    question_map = {qa['id']: qa.get('question', '')      for qa in qa_pairs}
    group_map_qa = {qa['id']: qa.get('group', 'unknown')  for qa in qa_pairs}

    # Aggregate group statistics
    group_stats: dict = {}
    for r in per:
        g = r.get('group', 'unknown')
        group_stats.setdefault(g, []).append(r)

    lines = []
    def ln(s=''):
        lines.append(s)

    def hr():
        ln('---')
        ln()

    # ═══════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════
    ln('# 📊 RAG Evaluation Report')
    ln()
    ln('> Báo cáo tự động từ pipeline đánh giá RAG — **Luật Chuyển giao Công nghệ**')
    ln()
    ln('| | |')
    ln('|---|---|')
    ln(f'| **Run ID** | `{rid}` |')
    ln(f'| **Generated** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |')
    ln(f'| **Strategy** | `{cfg.get("strategy", "?")}` |')
    ln(f'| **Top-K / Rerank-K** | {cfg.get("top_k", "?")} / {cfg.get("rerank_top_k", "?")} |')
    ln(f'| **Temperature** | {cfg.get("temperature", "?")} |')
    ln(f'| **Judge Model** | `{cfg.get("judge_model", "same_as_generator")}` |')
    ln(f'| **Test Cases** | {len(per)} |')
    ln()

    # ── Model consistency warnings ────────────────────────────────
    unexpected_swap   = [r['test_id'] for r in per
                         if not r['group'].startswith('D')
                         and r.get('model_consistent', True) is True]
    bias_test_invalid = [r['test_id'] for r in per
                         if r['group'].startswith('D')
                         and r.get('model_consistent', True) is False]

    if unexpected_swap:
        ln('> [!WARNING]')
        ln(f'> **Model Swap Không Mong Muốn** — Judge = Generator tại: `{", ".join(unexpected_swap)}`  ')
        ln('> Kết quả Factual Consistency của các case này có thể bị self-bias.')
        ln()
    if bias_test_invalid:
        ln('> [!WARNING]')
        ln(f'> **Bias Test Không Hợp Lệ** — OpenRouter swap model tại D* cases: `{", ".join(bias_test_invalid)}`  ')
        ln('> Các case này không thể dùng để đo self-bias.')
        ln()

    hr()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: EXECUTIVE SUMMARY — SCORECARD
    # ═══════════════════════════════════════════════════════════════
    ln('## 1. Executive Summary — Scorecard')
    ln()
    ln('> 🟢 ≥ 0.80 · 🟡 ≥ 0.50 · 🔴 < 0.50 · ⚪ N/A')
    ln()

    t1r_agg = agg['tier1_retriever']
    t1k_agg = agg['tier1_reranker']
    t1g_agg = agg['tier1_generator']
    t2_agg  = agg['tier2_pipeline']

    # Retriever block
    hr5  = t1r_agg.get('hit_rate@5')
    crc  = t1r_agg.get('context_recall')
    mrr  = t1k_agg.get('mrr')
    cpc  = t1k_agg.get('context_precision')
    fth  = t1g_agg.get('faithfulness')
    anr  = t1g_agg.get('answer_relevancy')
    fct  = t1g_agg.get('factual_consistency')
    sem  = t2_agg.get('semantic_similarity_mean')
    lat  = t2_agg.get('e2e_latency_mean_ms')
    tok  = t2_agg.get('tokens_per_query_mean')

    ln('### 🔍 Retriever')
    ln()
    ln(f'| Metric | Score | Status |')
    ln(f'|--------|------:|--------|')
    ln(f'| Hit Rate@5 *(ranx)* | **{fmt(hr5, 3)}** | {score_badge(hr5)} |')
    ln(f'| Context Recall *(RAGAs)* | **{fmt(crc, 3)}** | {score_badge(crc)} |')
    ln()

    ln('### 📐 Reranker')
    ln()
    ln(f'| Metric | Score | Status |')
    ln(f'|--------|------:|--------|')
    ln(f'| MRR *(ranx)* | **{fmt(mrr, 3)}** | {score_badge(mrr)} |')
    ln(f'| Context Precision *(RAGAs)* | **{fmt(cpc, 3)}** | {score_badge(cpc)} |')
    ln()

    ln('### ✍️ Generator')
    ln()
    ln(f'| Metric | Score | Status |')
    ln(f'|--------|------:|--------|')
    ln(f'| Faithfulness *(RAGAs)* | **{fmt(fth, 3)}** | {score_badge(fth)} |')
    ln(f'| Answer Relevancy *(RAGAs)* | **{fmt(anr, 3)}** | {score_badge(anr)} |')
    ln(f'| Factual Consistency *(prod HCheck)* | **{fmt(fct, 3)}** | {score_badge(fct)} |')
    ln()

    ln('### 🌐 End-to-End Pipeline')
    ln()
    ln(f'| Metric | Value | Status |')
    ln(f'|--------|------:|--------|')
    ln(f'| Semantic Similarity | **{fmt(sem, 3)}** | {score_badge(sem)} |')
    ln(f'| Avg E2E Latency | **{ms_to_s(lat)}** | — |')
    ln(f'| Avg Tokens / Query | **{fmt(tok, 0)}** | — |')
    ln()

    hr()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: PER-GROUP BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    ln('## 2. Kết Quả Theo Nhóm (Group Breakdown)')
    ln()
    ln('> Mỗi nhóm test case được thiết kế để kiểm tra một failure mode cụ thể của pipeline.')
    ln()

    # Build group-level aggregate
    ln('| Group | N | Hit@5 | CtxRec | CtxPre | Faith | AnsRel | SemSim | HCheck |')
    ln('|-------|--:|------:|-------:|-------:|------:|-------:|-------:|-------:|')

    all_groups = sorted(group_stats.keys())
    for g in all_groups:
        rs = group_stats[g]
        n  = len(rs)
        def gavg(fn):
            vals = [fn(r) for r in rs]
            return safe_mean(vals)

        g_hr5  = gavg(lambda r: r.get('hit_rate_5'))
        g_crc  = gavg(lambda r: r['tier1_retriever']['context_recall'])
        g_cpc  = gavg(lambda r: r['tier1_reranker']['context_precision'])
        g_fth  = gavg(lambda r: r['tier1_generator']['faithfulness'])
        g_anr  = gavg(lambda r: r['tier1_generator']['answer_relevancy'])
        g_sem  = gavg(lambda r: r['tier2']['semantic_similarity'])
        g_hck  = gavg(lambda r: r['tier1_generator']['factual_consistency'])

        ln(
            f'| `{g}` | {n} '
            f'| {score_badge(g_hr5)}{fmt(g_hr5, 2)} '
            f'| {score_badge(g_crc)}{fmt(g_crc, 2)} '
            f'| {score_badge(g_cpc)}{fmt(g_cpc, 2)} '
            f'| {score_badge(g_fth)}{fmt(g_fth, 2)} '
            f'| {score_badge(g_anr)}{fmt(g_anr, 2)} '
            f'| {score_badge(g_sem)}{fmt(g_sem, 2)} '
            f'| {score_badge(g_hck)}{fmt(g_hck, 2)} |'
        )
    ln()

    hr()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: MERGED PER-CASE TABLE (metrics + latency + isolated)
    # ═══════════════════════════════════════════════════════════════
    ln('## 3. Bảng Chi Tiết Từng Test Case')
    ln()
    ln('> **Chú giải metric:** 🟢 ≥ 0.80 · 🟡 ≥ 0.50 · 🔴 < 0.50  ')
    ln('> **Chú giải cột:** `_iso` = component isolated (generator nhận golden context,')
    ln('> reranker nhận pool = retrieved ∪ golden). `E2E` = toàn bộ chain. `—` = chưa chạy.')
    ln()

    # Header: base metrics + latency + tokens + isolated
    ln('| ID | Group | Hit@5 | CtxRec | CtxPre | Faith | AnsRel | HCheck | SemSim '
       '| E2E | Tokens '
       '| Faith_iso | AnsRel_iso | CtxPre_iso | MRR_iso |')
    ln('|----|-------|------:|-------:|-------:|------:|-------:|-------:|-------:'
       '|----:|-------:'
       '|----------:|-----------:|-----------:|--------:|')

    for r in per:
        tid = r['test_id']
        t1r = r['tier1_retriever']
        t1k = r['tier1_reranker']
        t1g = r['tier1_generator']
        t2  = r['tier2']
        gr  = gen_per_map.get(tid, {})
        rr  = rerank_per_map.get(tid, {})

        # HCheck cell (parse error flag)
        hck_val  = t1g.get('factual_consistency')
        hck_perr = t1g.get('hcheck_parse_error', False)
        hck_cell = f'🔧{fmt(hck_val, 2)}' if hck_perr else f'{score_badge(hck_val)}{fmt(hck_val, 2)}'

        def c(v):
            return f'{score_badge(v)}{fmt(v, 2)}' if v is not None else '—'

        # Isolated metrics (None → '—')
        faith_iso  = gr.get('faithfulness_iso')
        ansrel_iso = gr.get('answer_rel_iso')
        ctxpre_iso = rr.get('context_precision_iso')
        mrr_iso    = rr.get('mrr_iso')

        ln(
            f'| {tid} | `{r["group"]}` '
            f'| {c(r.get("hit_rate_5"))} '
            f'| {c(t1r["context_recall"])} '
            f'| {c(t1k["context_precision"])} '
            f'| {c(t1g["faithfulness"])} '
            f'| {c(t1g["answer_relevancy"])} '
            f'| {hck_cell} '
            f'| {c(t2["semantic_similarity"])} '
            f'| {ms_to_s(t2.get("e2e_latency_ms"))} '
            f'| {fmt(t2.get("tokens"), 0)} '
            f'| {c(faith_iso)} '
            f'| {c(ansrel_iso)} '
            f'| {c(ctxpre_iso)} '
            f'| {c(mrr_iso)} |'
        )
    ln()

    # Note isolated availability
    if not (gen_metrics or rerank_metrics):
        ln('> _Cột `_iso` trống vì chưa chạy component isolated eval. Chạy:_')
        ln('> ```bash')
        ln('> python -m eval.phase2_collect.generator')
        ln('> python -m eval.phase2_collect.reranker')
        ln('> python -m eval.phase3_compute.compute_generator')
        ln('> python -m eval.phase3_compute.compute_reranker')
        ln('> ```')
        ln()

        # ── Save ──────────────────────────────────────────────────────
    report_path = os.path.join(
        RESULTS_DIR,
        f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[report] Report saved: {report_path}')
    return report_path


if __name__ == '__main__':
    run()
