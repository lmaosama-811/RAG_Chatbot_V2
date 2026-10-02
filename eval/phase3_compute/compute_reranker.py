"""
eval/phase3_compute/compute_reranker.py
=========================================
PHASE 3 (Component): Tính metrics cho Reranker Isolated.

INPUT  : eval/results/reranker_collected_<timestamp>.json (file mới nhất)
OUTPUT : eval/results/reranker_metrics_<timestamp>.json

KHÔNG gọi LLM — chỉ đọc JSON và tính toán.
Có thể chạy nhiều lần tùy ý.

Metrics tính:
    context_precision_iso : RAGAs (đã tính trong reranker.py, đọc lại)
    mrr_iso               : 1/rank của golden chunk đầu tiên trong top-K
    hit_rate_iso@5        : có golden chunk nào trong top-5 không?
    golden_in_pool_rate   : tỷ lệ test case mà golden có trong pool

Chạy:
    python -m eval.phase3_compute.compute_reranker
    python -m eval.phase3_compute.compute_reranker --collected results/reranker_collected_XYZ.json
"""
import sys, os, json, glob, argparse, statistics
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from eval.config import RESULTS_DIR


def find_latest_reranker_collected(results_dir: str) -> str:
    files = glob.glob(os.path.join(results_dir, 'reranker_collected_*.json'))
    if not files:
        raise FileNotFoundError(
            f'Không tìm thấy reranker_collected_*.json trong {results_dir}\n'
            f'Chạy trước: python -m eval.phase2_collect.reranker'
        )
    return max(files, key=os.path.getmtime)


def safe_mean(values: list):
    valid = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return round(statistics.mean(valid), 4) if valid else None


def compute_mrr(reranked: list, golden_chunk_ids: list) -> float | None:
    """
    MRR = 1/rank của golden chunk đầu tiên xuất hiện trong reranked list.
    Rank bắt đầu từ 1.
    Trả về 0.0 nếu không có golden chunk nào trong top-K.
    """
    if not golden_chunk_ids:
        return None
    golden_set = set(golden_chunk_ids)
    for item in reranked:
        if item['chunk_id'] in golden_set:
            return round(1.0 / item['rank'], 4)
    return 0.0


def compute_hit_rate(reranked: list, golden_chunk_ids: list, k: int = 5) -> float | None:
    """
    Hit Rate@k = 1.0 nếu ít nhất 1 golden chunk trong top-k, else 0.0.
    """
    if not golden_chunk_ids:
        return None
    top_k_ids = {item['chunk_id'] for item in reranked[:k]}
    return 1.0 if top_k_ids & set(golden_chunk_ids) else 0.0


def run(collected_path: str = None):
    if collected_path is None:
        collected_path = find_latest_reranker_collected(RESULTS_DIR)

    print(f'[compute_reranker] Input : {os.path.basename(collected_path)}')
    data    = json.load(open(collected_path, encoding='utf-8'))
    results = data['results']

    ok_results  = [r for r in results if 'error' not in r]
    err_results = [r for r in results if 'error' in r]

    print(f'[compute_reranker] Total: {len(results)} | OK: {len(ok_results)} | Error: {len(err_results)}')

    # ── Per-case metrics ─────────────────────────────────────────────────────
    per_case      = []
    ctx_pre_scores  = []
    mrr_scores      = []
    hit_rate_scores = []
    pool_had_golden_count = 0
    golden_added_count    = 0
    rerank_latencies      = []
    ragas_latencies       = []

    for r in ok_results:
        golden_ids  = r.get('golden_chunk_ids', [])
        reranked    = r.get('reranked_isolated', [])
        ctx_pre     = r.get('ragas_reranker_isolated', {}).get('context_precision')
        pool_had    = r.get('pool_had_golden', False)
        golden_add  = r.get('golden_added_to_pool', False)
        t_rerank    = r.get('timings_ms', {}).get('reranking')
        t_ragas     = r.get('timings_ms', {}).get('ragas')

        mrr      = compute_mrr(reranked, golden_ids)
        hit_rate = compute_hit_rate(reranked, golden_ids, k=5)

        ctx_pre_scores.append(ctx_pre)
        mrr_scores.append(mrr)
        hit_rate_scores.append(hit_rate)
        if pool_had:
            pool_had_golden_count += 1
        if golden_add:
            golden_added_count += 1
        if t_rerank is not None:
            rerank_latencies.append(t_rerank)
        if t_ragas is not None:
            ragas_latencies.append(t_ragas)

        per_case.append({
            'test_id':              r['test_id'],
            'group':                r.get('group', ''),
            'pool_size':            r.get('pool_size', 0),
            'pool_had_golden':      pool_had,
            'golden_added_to_pool': golden_add,
            'context_precision_iso': ctx_pre,
            'mrr_iso':              mrr,
            'hit_rate_iso_at5':     hit_rate,
            'top5_chunk_ids':       [item['chunk_id'] for item in reranked[:5]],
            'golden_chunk_ids':     golden_ids,
            'rerank_latency_ms':    t_rerank,
            'ragas_latency_ms':     t_ragas,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(ok_results)
    aggregate = {
        'total':                      len(results),
        'ok':                         n,
        'errors':                     len(err_results),
        'pool_had_golden_rate':       round(pool_had_golden_count / n, 4) if n else None,
        'golden_added_rate':          round(golden_added_count / n, 4) if n else None,
        'context_precision_iso_mean': safe_mean(ctx_pre_scores),
        'mrr_iso_mean':               safe_mean(mrr_scores),
        'hit_rate_iso_at5_mean':      safe_mean(hit_rate_scores),
        'rerank_latency_mean_ms':     safe_mean(rerank_latencies),
        'ragas_latency_mean_ms':      safe_mean(ragas_latencies),
    }

    # ── Output ────────────────────────────────────────────────────────────────
    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'reranker_metrics_{run_id}.json')

    output = {
        'generated_at':     run_id,
        'source_collected': os.path.basename(collected_path),
        'aggregate':        aggregate,
        'per_case':         per_case,
        'errors':           [{'test_id': r['test_id'], 'error': r['error']}
                             for r in err_results],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('[compute_reranker] ─── Aggregate ────────────────────────────')
    print(f'  pool_had_golden_rate        : {aggregate["pool_had_golden_rate"]}')
    print(f'  golden_added_rate           : {aggregate["golden_added_rate"]}')
    print(f'  context_precision_iso_mean  : {aggregate["context_precision_iso_mean"]}')
    print(f'  mrr_iso_mean                : {aggregate["mrr_iso_mean"]}')
    print(f'  hit_rate_iso@5_mean         : {aggregate["hit_rate_iso_at5_mean"]}')
    print(f'[compute_reranker] Output: {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collected', type=str, default=None)
    args = parser.parse_args()
    run(collected_path=args.collected)
