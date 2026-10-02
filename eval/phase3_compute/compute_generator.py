"""
eval/phase3_compute/compute_generator.py
==========================================
PHASE 3 (Component): Tính metrics cho Generator Isolated.

INPUT  : eval/results/generator_collected_<timestamp>.json (file mới nhất)
OUTPUT : eval/results/generator_metrics_<timestamp>.json

KHÔNG gọi LLM — chỉ đọc JSON và tính toán.
Có thể chạy nhiều lần tùy ý.

Chạy:
    python -m eval.phase3_compute.compute_generator
    python -m eval.phase3_compute.compute_generator --collected results/generator_collected_XYZ.json
"""
import sys, os, json, glob, argparse, statistics
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from eval.config import RESULTS_DIR


def find_latest_generator_collected(results_dir: str) -> str:
    files = glob.glob(os.path.join(results_dir, 'generator_collected_*.json'))
    if not files:
        raise FileNotFoundError(
            f'Không tìm thấy generator_collected_*.json trong {results_dir}\n'
            f'Chạy trước: python -m eval.phase2_collect.generator'
        )
    return max(files, key=os.path.getmtime)


def safe_mean(values: list):
    valid = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return round(statistics.mean(valid), 4) if valid else None


def run(collected_path: str = None):
    if collected_path is None:
        collected_path = find_latest_generator_collected(RESULTS_DIR)

    print(f'[compute_generator] Input : {os.path.basename(collected_path)}')
    data = json.load(open(collected_path, encoding='utf-8'))
    results = data['results']

    ok_results  = [r for r in results if 'error' not in r]
    err_results = [r for r in results if 'error' in r]

    print(f'[compute_generator] Total: {len(results)} | OK: {len(ok_results)} | Error: {len(err_results)}')

    # ── Per-case metrics ─────────────────────────────────────────────────────
    per_case = []
    faith_scores  = []
    ansrel_scores = []
    found_count   = 0
    gen_latencies = []
    ragas_latencies = []

    for r in ok_results:
        ragas_iso = r.get('ragas_isolated', {})
        faith     = ragas_iso.get('faithfulness')
        ansrel    = ragas_iso.get('answer_relevancy')
        found     = r.get('golden_chunks_found', False)
        t_gen     = r.get('timings_ms', {}).get('generation')
        t_ragas   = r.get('timings_ms', {}).get('ragas')

        faith_scores.append(faith)
        ansrel_scores.append(ansrel)
        if found:
            found_count += 1
        if t_gen is not None:
            gen_latencies.append(t_gen)
        if t_ragas is not None:
            ragas_latencies.append(t_ragas)

        per_case.append({
            'test_id':          r['test_id'],
            'group':            r.get('group', ''),
            'golden_found':     found,
            'faithfulness_iso': faith,
            'answer_rel_iso':   ansrel,
            'gen_latency_ms':   t_gen,
            'ragas_latency_ms': t_ragas,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregate = {
        'total':                    len(results),
        'ok':                       len(ok_results),
        'errors':                   len(err_results),
        'golden_chunks_found_rate': round(found_count / len(ok_results), 4) if ok_results else None,
        'faithfulness_iso_mean':    safe_mean(faith_scores),
        'answer_rel_iso_mean':      safe_mean(ansrel_scores),
        'gen_latency_mean_ms':      safe_mean(gen_latencies),
        'ragas_latency_mean_ms':    safe_mean(ragas_latencies),
    }

    # ── Output ────────────────────────────────────────────────────────────────
    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'generator_metrics_{run_id}.json')

    output = {
        'generated_at':      run_id,
        'source_collected':  os.path.basename(collected_path),
        'aggregate':         aggregate,
        'per_case':          per_case,
        'errors':            [{'test_id': r['test_id'], 'error': r['error']}
                              for r in err_results],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('[compute_generator] ─── Aggregate ───────────────────────────')
    print(f'  golden_chunks_found_rate : {aggregate["golden_chunks_found_rate"]}')
    print(f'  faithfulness_iso_mean    : {aggregate["faithfulness_iso_mean"]}')
    print(f'  answer_rel_iso_mean      : {aggregate["answer_rel_iso_mean"]}')
    print(f'  gen_latency_mean_ms      : {aggregate["gen_latency_mean_ms"]}')
    print(f'[compute_generator] Output: {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collected', type=str, default=None)
    args = parser.parse_args()
    run(collected_path=args.collected)
