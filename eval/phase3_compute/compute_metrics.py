"""
eval/phase3_compute/compute_metrics.py
=========================================
PHASE 3: Tính tất cả metrics từ collected_outputs.json.

INPUT  : eval/results/collected_YYYYMMDD.json  (file mới nhất tự động)
         eval/data/qa_pairs.json               (để lấy golden_chunk_ids)
OUTPUT : eval/results/metrics_YYYYMMDD.json

Chạy lệnh:
    python3 -m eval.phase3_compute.compute_metrics
    python3 -m eval.phase3_compute.compute_metrics --collected results/collected_20260919.json

KHÔNG gọi LLM, KHÔNG gọi API — chỉ đọc JSON và tính toán.
Có thể chạy nhiều lần tùy ý mà không tốn thêm chi phí.

Metrics được tính:
    RETRIEVER:
        Hit Rate@5     (ranx)       : ≥1 golden chunk trong top-5 retrieval?
        Context Recall (RAGAs saved): context cover đủ golden_answer không?

    RERANKER:
        MRR            (ranx)       : rank trung bình của golden chunk sau rerank
        Context Precision (RAGAs)   : reranked context có chất lượng không?

    GENERATOR:
        Faithfulness      (RAGAs)   : answer có grounded trong context không?
        Answer Relevancy  (RAGAs)   : answer có trả lời đúng câu hỏi không?
        Factual Consistency (prod)  : hallucination_check.confidence

    PIPELINE (Tier 2):
        Semantic Similarity : cosine similarity(generated_answer, golden_answer)
        E2E Latency         : tổng thời gian pipeline (không tính RAGAs eval)
        Tokens per Query    : tổng token ước tính (tiktoken)
"""
import sys, os, json, glob, argparse
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from eval.config import QA_PAIRS_PATH, RESULTS_DIR
from eval.utils.metrics import build_qrels, build_run_from_stage, SemSimilarityScorer

from ranx import evaluate as ranx_evaluate
import statistics


def find_latest_collected(results_dir: str) -> str:
    """Tìm file collected_*.json mới nhất."""
    files = glob.glob(os.path.join(results_dir, 'collected_*.json'))
    if not files:
        raise FileNotFoundError(
            f'Không tìm thấy file collected_*.json trong: {results_dir}\n'
            f'Chạy phase2 trước: python3 -m eval.phase2_collect.collect'
        )
    return max(files, key=os.path.getmtime)


def safe_mean(values: list) -> float:
    """Tính mean, bỏ qua None và NaN."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return round(statistics.mean(valid), 4) if valid else None


def safe_variance(values: list) -> float:
    """Tính variance, cần ít nhất 2 giá trị hợp lệ."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return round(statistics.variance(valid), 6) if len(valid) >= 2 else None


def compute_per_case_hit_rate(ok_results: list, qa_pairs: list, k: int = 5) -> dict:
    """
    Tính hit_rate@k cho từng query đơn lẻ.
    Không dùng ranx (ranx chỉ cho aggregate) — tính thủ công từ collected JSON.

    OUTPUT: {test_id: 1 hoặc 0 hoặc None (nếu không có golden_chunk_ids)}
    """
    golden_by_id = {qa['id']: set(qa.get('golden_chunk_ids', [])) for qa in qa_pairs}
    result = {}
    for r in ok_results:
        tid    = r['test_id']
        golden = golden_by_id.get(tid, set())
        if not golden:
            result[tid] = None   # Không có golden → không tính
            continue
        retrieved = [
            c['chunk_id']
            for c in r.get('step_outputs', {}).get('retrieval_final', [])
        ]
        top_k     = retrieved[:k]
        result[tid] = 1 if any(cid in golden for cid in top_k) else 0
    return result


def compute_per_case_mrr(ok_results: list, qa_pairs: list) -> dict:
    """
    Tính MRR cho từng query đơn lẻ từ reranking stage.
    MRR = 1/rank của golden chunk đầu tiên trong danh sách reranked.
    Nếu không tìm thấy golden chunk nào → MRR = 0.

    OUTPUT: {test_id: float (0.0 – 1.0) hoặc None}
    """
    golden_by_id = {qa['id']: set(qa.get('golden_chunk_ids', [])) for qa in qa_pairs}
    result = {}
    for r in ok_results:
        tid    = r['test_id']
        golden = golden_by_id.get(tid, set())
        if not golden:
            result[tid] = None
            continue
        reranked = [
            c['chunk_id']
            for c in r.get('step_outputs', {}).get('reranking', [])
        ]
        mrr = 0.0
        for rank, cid in enumerate(reranked, 1):
            if cid in golden:
                mrr = round(1.0 / rank, 4)
                break
        result[tid] = mrr
    return result


# Ngưỡng để classify E1/E2:
#   E1: component đều OK nhưng pipeline fail
#   E2: có ít nhất 1 component fail nhưng pipeline pass
COMPONENT_PASS_THRESHOLD = 0.5   # context_recall, context_precision, faithfulness, answer_relevancy
PIPELINE_PASS_THRESHOLD  = 0.65  # semantic_similarity


def classify_e_type(per_case_record: dict) -> str:
    """
    Phân loại 1 test case thuộc dạng nào:
      'E1' : tất cả component metrics cao nhưng pipeline fail (SemanticSim thấp)
      'E2' : ít nhất 1 component metric thấp nhưng pipeline pass (SemanticSim cao)
      None : bình thường hoặc thiếu data

    NOTE: Chức năng này để VALIDATE các test cases được label là E1/E2.
    Nếu 1 case được label là E1 nhưng compute không detect ra E1 → cần xem lại.
    """
    t1r = per_case_record.get('tier1_retriever', {})
    t1k = per_case_record.get('tier1_reranker', {})
    t1g = per_case_record.get('tier1_generator', {})
    t2  = per_case_record.get('tier2', {})

    hr5   = per_case_record.get('hit_rate_5')   # 0 or 1 or None
    cr    = t1r.get('context_recall')
    cp    = t1k.get('context_precision')
    faith = t1g.get('faithfulness')
    rel   = t1g.get('answer_relevancy')
    semsim = t2.get('semantic_similarity')

    if semsim is None:
        return None

    # Component metrics: sử dụng những cái có data
    component_vals = [v for v in [cr, cp, faith, rel] if v is not None]
    # hit_rate_5 là binary — treat 0 = fail
    if hr5 is not None:
        component_vals.append(float(hr5))

    if not component_vals:
        return None

    all_component_pass = all(v >= COMPONENT_PASS_THRESHOLD for v in component_vals)
    any_component_fail = any(v < COMPONENT_PASS_THRESHOLD for v in component_vals)
    pipeline_pass      = semsim >= PIPELINE_PASS_THRESHOLD

    if all_component_pass and not pipeline_pass:
        return 'E1'
    if any_component_fail and pipeline_pass:
        return 'E2'
    return None


def compute(collected_path: str = None):
    # ── Load data ───────────────────────────────────────────────────────────
    if collected_path is None:
        collected_path = find_latest_collected(RESULTS_DIR)
    print(f'[compute] Using: {collected_path}')

    with open(collected_path, 'r', encoding='utf-8') as f:
        collected = json.load(f)

    with open(QA_PAIRS_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    results      = collected['results']
    # Lọc bỏ error records
    ok_results   = [r for r in results if 'error' not in r]
    run_id       = collected.get('run_id', 'unknown')

    print(f'[compute] {len(ok_results)}/{len(results)} test cases hợp lệ')

    # ── Tier 1 — Retriever ───────────────────────────────────────────────────
    print('[compute] Tính Hit Rate@5 (ranx)...')
    qrels         = build_qrels(qa_pairs)
    run_retrieval = build_run_from_stage(ok_results, stage='retrieval')

    # Chỉ tính trên những queries có cả qrels lẫn run
    retriever_ranx = {}
    if run_retrieval and qrels:
        try:
            _res = ranx_evaluate(qrels, run_retrieval, ['hit_rate@5'])
            retriever_ranx = _res if isinstance(_res, dict) else {'hit_rate@5': float(_res)}
        except Exception as e:
            print(f'[WARN] ranx hit_rate@5 failed: {e}')

    context_recalls = [
        r['step_outputs']['ragas'].get('context_recall')
        for r in ok_results
        if 'ragas' in r.get('step_outputs', {})
    ]

    # ── Tier 1 — Reranker ────────────────────────────────────────────────────
    print('[compute] Tính MRR (ranx)...')
    run_reranker = build_run_from_stage(ok_results, stage='reranking')

    reranker_ranx = {}
    if run_reranker and qrels:
        try:
            _res = ranx_evaluate(qrels, run_reranker, ['mrr'])
            reranker_ranx = _res if isinstance(_res, dict) else {'mrr': float(_res)}
        except Exception as e:
            print(f'[WARN] ranx mrr failed: {e}')

    context_precisions = [
        r['step_outputs']['ragas'].get('context_precision')
        for r in ok_results
        if 'ragas' in r.get('step_outputs', {})
    ]

    # ── Tier 1 — Generator ───────────────────────────────────────────────────
    faithfulness_scores = [
        r['step_outputs']['ragas'].get('faithfulness')
        for r in ok_results if 'ragas' in r.get('step_outputs', {})
    ]
    answer_relevancy_scores = [
        r['step_outputs']['ragas'].get('answer_relevancy')
        for r in ok_results if 'ragas' in r.get('step_outputs', {})
    ]
    factual_consistency_scores = [
        r['step_outputs']['hallucination_check'].get('confidence')
        for r in ok_results if 'hallucination_check' in r.get('step_outputs', {})
    ]

    # ── Tier 2 — Pipeline ────────────────────────────────────────────────────
    print('[compute] Tính Semantic Similarity (local, no API)...')
    scorer = SemSimilarityScorer()
    sem_sim_pairs = [
        (r['step_outputs']['generated_answer'], r['golden_answer'])
        for r in ok_results
        if r.get('golden_answer') and r.get('step_outputs', {}).get('generated_answer')
    ]
    sem_sim_scores = scorer.batch_score(sem_sim_pairs) if sem_sim_pairs else []

    e2e_latencies  = [r['timings_ms']['total_e2e']      for r in ok_results if 'timings_ms' in r]
    tokens_per_q   = [r['tokens_estimated']['total']     for r in ok_results if 'tokens_estimated' in r]
    model_inconsistent = [r['test_id'] for r in ok_results
                          if not r.get('models_used', {}).get('model_consistent', True)]

    # ── Per-case ranx metrics (tính thủ công, không qua ranx aggregate) ────────
    print('[compute] Tính per-case Hit Rate@5 và MRR...')
    per_case_hit_rate = compute_per_case_hit_rate(ok_results, qa_pairs, k=5)
    per_case_mrr      = compute_per_case_mrr(ok_results, qa_pairs)

    # ── D1 variance (self-consistency bias) ─────────────────────────────────
    d1_results = [r for r in ok_results if r.get('group', '').startswith('D1')]
    d1_fc_scores = [
        r.get('step_outputs', {}).get('hallucination_check', {}).get('confidence')
        for r in d1_results
    ]
    d1_gen_answers = [
        r.get('step_outputs', {}).get('generated_answer', '')
        for r in d1_results
    ]

    # ── Build per-test-case metrics ─────────────────────────────────────────
    print('[compute] Building per-test-case metrics...')

    # Map test_id → sem_sim score
    sem_sim_map = {}
    for (gen, gold), score in zip(sem_sim_pairs, sem_sim_scores):
        for r in ok_results:
            if r.get('step_outputs', {}).get('generated_answer') == gen:
                sem_sim_map[r['test_id']] = round(score, 4)

    per_case = []
    for r in ok_results:
        tid   = r['test_id']
        ragas = r.get('step_outputs', {}).get('ragas', {})
        hck   = r.get('step_outputs', {}).get('hallucination_check', {})

        record = {
            'test_id':  tid,
            'group':    r.get('group', 'unknown'),
            'model_consistent': r.get('models_used', {}).get('model_consistent', None),
            # Per-case ranx metrics — cần cho E2 detection
            'hit_rate_5': per_case_hit_rate.get(tid),
            'mrr':        per_case_mrr.get(tid),
            'tier1_retriever': {
                'context_recall': ragas.get('context_recall'),
            },
            'tier1_reranker': {
                'context_precision': ragas.get('context_precision'),
            },
            'tier1_generator': {
                'faithfulness':        ragas.get('faithfulness'),
                'answer_relevancy':    ragas.get('answer_relevancy'),
                'factual_consistency': hck.get('confidence'),
            },
            'tier2': {
                'semantic_similarity': sem_sim_map.get(tid),
                'e2e_latency_ms':     r.get('timings_ms', {}).get('total_e2e'),
                'tokens':             r.get('tokens_estimated', {}).get('total'),
                # Per-step latency breakdown (ms)
                'latency_analyze_ms':  r.get('timings_ms', {}).get('analyze_query'),
                'latency_gen_ms':      r.get('timings_ms', {}).get('generation'),
                'latency_hcheck_ms':   r.get('timings_ms', {}).get('hallucination_check'),
                'latency_rerank_ms':   r.get('timings_ms', {}).get('reranking'),
                # Token breakdown (estimated)
                'tok_analyze_in':  r.get('tokens_estimated', {}).get('analyze_in'),
                'tok_analyze_out': r.get('tokens_estimated', {}).get('analyze_out'),
                'tok_gen_in':      r.get('tokens_estimated', {}).get('gen_in'),
                'tok_gen_out':     r.get('tokens_estimated', {}).get('gen_out'),
                'tok_hcheck_in':   r.get('tokens_estimated', {}).get('hcheck_in'),
                'tok_hcheck_out':  r.get('tokens_estimated', {}).get('hcheck_out'),
            },
        }
        # Auto-classify E1/E2 để validate labeled cases
        record['detected_e_type'] = classify_e_type(record)
        per_case.append(record)


    # ── Build aggregate ───────────────────────────────────────────────────────
    _d1_valid = [v for v in d1_fc_scores if v is not None]
    aggregate = {
        'tier1_retriever': {
            'hit_rate@5':     round(float(retriever_ranx.get('hit_rate@5', 0)), 4),
            'context_recall': safe_mean(context_recalls),
        },
        'tier1_reranker': {
            'mrr':               round(float(reranker_ranx.get('mrr', 0)), 4),
            'context_precision': safe_mean(context_precisions),
        },
        'tier1_generator': {
            'faithfulness':        safe_mean(faithfulness_scores),
            'answer_relevancy':    safe_mean(answer_relevancy_scores),
            'factual_consistency': safe_mean(factual_consistency_scores),
        },
        'tier2_pipeline': {
            'semantic_similarity_mean': safe_mean(sem_sim_scores),
            'e2e_latency_mean_ms':      safe_mean(e2e_latencies),
            'tokens_per_query_mean':    safe_mean(tokens_per_q),
            # Per-step latency averages
            'latency_analyze_mean_ms':  safe_mean([r.get('timings_ms', {}).get('analyze_query')         for r in ok_results if r.get('timings_ms', {}).get('analyze_query')         is not None]),
            'latency_gen_mean_ms':      safe_mean([r.get('timings_ms', {}).get('generation')            for r in ok_results if r.get('timings_ms', {}).get('generation')            is not None]),
            'latency_hcheck_mean_ms':   safe_mean([r.get('timings_ms', {}).get('hallucination_check')   for r in ok_results if r.get('timings_ms', {}).get('hallucination_check')   is not None]),
            'latency_rerank_mean_ms':   safe_mean([r.get('timings_ms', {}).get('reranking')             for r in ok_results if r.get('timings_ms', {}).get('reranking')             is not None]),
            # Per-step token averages
            'tok_analyze_in_mean':  safe_mean([r.get('tokens_estimated', {}).get('analyze_in')  for r in ok_results if r.get('tokens_estimated', {}).get('analyze_in')  is not None]),
            'tok_analyze_out_mean': safe_mean([r.get('tokens_estimated', {}).get('analyze_out') for r in ok_results if r.get('tokens_estimated', {}).get('analyze_out') is not None]),
            'tok_gen_in_mean':      safe_mean([r.get('tokens_estimated', {}).get('gen_in')      for r in ok_results if r.get('tokens_estimated', {}).get('gen_in')      is not None]),
            'tok_gen_out_mean':     safe_mean([r.get('tokens_estimated', {}).get('gen_out')     for r in ok_results if r.get('tokens_estimated', {}).get('gen_out')     is not None]),
            'tok_hcheck_in_mean':   safe_mean([r.get('tokens_estimated', {}).get('hcheck_in')   for r in ok_results if r.get('tokens_estimated', {}).get('hcheck_in')   is not None]),
            'tok_hcheck_out_mean':  safe_mean([r.get('tokens_estimated', {}).get('hcheck_out')  for r in ok_results if r.get('tokens_estimated', {}).get('hcheck_out')  is not None]),
        },
        'model_consistency': {
            'inconsistent_cases': model_inconsistent,
            'total_inconsistent': len(model_inconsistent),
        },
        # D1 self-consistency: cùng câu hỏi chạy 5 lần, đo variance của judge
        'd1_self_consistency': {
            'n_runs':                       len(_d1_valid),
            'factual_consistency_mean':     safe_mean(d1_fc_scores),
            'factual_consistency_variance': safe_variance(d1_fc_scores),
            'factual_consistency_stdev':    round(statistics.stdev(_d1_valid), 4)
                                            if len(_d1_valid) >= 2 else None,
            'generated_answers_unique':     len(set(a for a in d1_gen_answers if a)),
            'scores_per_run':               _d1_valid,
        },
        # E1/E2 auto-detection (để validate labeled cases)
        'e_type_detection': {
            'detected_E1':  [r['test_id'] for r in per_case if r.get('detected_e_type') == 'E1'],
            'detected_E2':  [r['test_id'] for r in per_case if r.get('detected_e_type') == 'E2'],
            'thresholds':   {'component_pass': COMPONENT_PASS_THRESHOLD,
                             'pipeline_pass':  PIPELINE_PASS_THRESHOLD},
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics_path = os.path.join(
        RESULTS_DIR,
        f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    output = {
        'run_id':        run_id,
        'collected_from': collected_path,
        'config':        collected.get('config', {}),
        'aggregate':     aggregate,
        'per_test_case': per_case,
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('AGGREGATE METRICS')
    print('=' * 60)
    print(f"  RETRIEVER:")
    print(f"    Hit Rate@5      : {aggregate['tier1_retriever']['hit_rate@5']}")
    print(f"    Context Recall  : {aggregate['tier1_retriever']['context_recall']}")
    print(f"  RERANKER:")
    print(f"    MRR             : {aggregate['tier1_reranker']['mrr']}")
    print(f"    Context Precision: {aggregate['tier1_reranker']['context_precision']}")
    print(f"  GENERATOR:")
    print(f"    Faithfulness    : {aggregate['tier1_generator']['faithfulness']}")
    print(f"    Answer Relevancy: {aggregate['tier1_generator']['answer_relevancy']}")
    print(f"    Factual Consist.: {aggregate['tier1_generator']['factual_consistency']}")
    print(f"  PIPELINE:")
    print(f"    Semantic Sim    : {aggregate['tier2_pipeline']['semantic_similarity_mean']}")
    print(f"    Avg E2E (ms)    : {aggregate['tier2_pipeline']['e2e_latency_mean_ms']}")
    print(f"    Avg Tokens/Q    : {aggregate['tier2_pipeline']['tokens_per_query_mean']}")
    if model_inconsistent:
        print(f"\n  ⚠ Model swap detected in: {model_inconsistent}")
    print('=' * 60)
    print(f'\n[compute] Metrics saved: {metrics_path}')
    print('[compute] Tiếp theo:')
    print('    python3 -m eval.phase4_report.generate_report')
    return metrics_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Phase 3: Compute Metrics')
    parser.add_argument(
        '--collected', type=str, default=None,
        help='Path to specific collected JSON file (default: latest)'
    )
    args = parser.parse_args()
    compute(collected_path=args.collected)
