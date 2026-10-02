"""
eval/phase2_collect/reranker.py
================================
PHASE 2 (Component Isolated): Reranker Eval

Chạy Reranker ở điều kiện isolated:
    Pool = retrieved_chunks (từ collected JSON) ∪ golden_chunks (nếu bị miss)
    → Reranker sắp xếp pool → RAGAs: context_precision_iso

Cần collect.py chạy trước để có collected JSON.

Output: eval/results/reranker_collected_<YYYYMMDD_HHMMSS>.json

Chạy:
    python -m eval.phase2_collect.reranker
    python -m eval.phase2_collect.reranker --collected results/collected_20260930.json
    python -m eval.phase2_collect.reranker --test-ids q001,q002
"""
import sys, os
import argparse
import json
import glob
import time
import random
import logging
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from app.model import embeddings, llm
from langchain_core.documents import Document

from eval.config import QA_PAIRS_PATH, CHUNKS_DUMP_PATH, RESULTS_DIR, RERANK_TOP_K
from eval.phase3_compute import ragas_wrapper
from eval.phase2_collect.collect import build_judge_llm, JUDGE_MODEL

# Dùng lại HierarchicalChunk từ app để rerank
from app.service.RAG_services.ChunkSplitters.HierarchicalChunk import HierarchicalChunk

for _n in ['google.ai.generativelanguage_v1beta', 'google.generativeai',
           'httpx', 'httpcore', 'langchain_core']:
    logging.getLogger(_n).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def find_latest_collected(results_dir: str) -> str:
    """Tìm collected_*.json mới nhất (không phải generator_collected hay reranker_collected)."""
    files = glob.glob(os.path.join(results_dir, 'collected_*.json'))
    # Loại trừ generator_collected và reranker_collected
    files = [f for f in files
             if os.path.basename(f).startswith('collected_')
             and 'generator' not in os.path.basename(f)
             and 'reranker' not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(
            f'Không tìm thấy collected_*.json trong {results_dir}\n'
            f'Chạy collect.py trước: python -m eval.phase2_collect.collect'
        )
    return max(files, key=os.path.getmtime)


def build_pool(qa: dict, retrieved_chunks: list, chunks_map: dict):
    """
    Pool = retrieved_chunks (E2E) ∪ golden_chunks.
    Đảm bảo golden chunks luôn có trong pool dù Retriever có miss hay không.

    Returns:
        (pool_docs, pool_had_golden, golden_added)
        - pool_docs: List[Document]
        - pool_had_golden: True nếu golden đã có sẵn trong retrieved
        - golden_added: True nếu phải thêm golden vào pool
    """
    golden_ids   = set(qa.get('golden_chunk_ids', []))
    retrieved_ids = {c['chunk_id'] for c in retrieved_chunks}

    pool_chunks  = list(retrieved_chunks)
    golden_added = False

    for gid in golden_ids:
        if gid not in retrieved_ids and gid in chunks_map:
            pool_chunks.append({
                'chunk_id': gid,
                'content':  chunks_map[gid]['content'],
            })
            golden_added = True

    pool_had_golden = golden_ids.issubset(retrieved_ids) if golden_ids else True

    # Convert sang Document list (shuffle để reranker không biết vị trí golden)
    pool_docs = [
        Document(
            page_content=c['content'],
            metadata={'chunk_id': c['chunk_id']}
        )
        for c in pool_chunks
    ]
    random.shuffle(pool_docs)

    return pool_docs, pool_had_golden, golden_added


def process_one(qa: dict, collected_result: dict, chunks_map: dict,
                chunker: HierarchicalChunk, judge_llm) -> dict:
    """
    Isolated Reranker eval cho 1 test case.
    Flow: build pool → rerank → RAGAs context_precision.
    """
    test_id          = qa['id']
    group            = qa.get('group', '')
    question         = qa['question']
    golden_answer    = qa.get('golden_answer', '')
    golden_chunk_ids = qa.get('golden_chunk_ids', [])

    so = collected_result.get('step_outputs', {})
    retrieved_chunks = so.get('retrieval_final', []) or []

    timings     = {}
    base_record = {
        'test_id':              test_id,
        'group':                group,
        'question':             question,
        'golden_answer':        golden_answer,
        'golden_chunk_ids':     golden_chunk_ids,
        'pool_size':            0,
        'pool_had_golden':      False,
        'golden_added_to_pool': False,
        'reranked_isolated':    [],
        'ragas_reranker_isolated': {'context_precision': None},
        'timings_ms':           {},
    }

    if not retrieved_chunks and not golden_chunk_ids:
        logger.warning(f'    [!] {test_id}: không có retrieved chunks và golden_chunk_ids → skip')
        return base_record

    # Build pool
    pool_docs, pool_had_golden, golden_added = build_pool(qa, retrieved_chunks, chunks_map)
    base_record['pool_size']            = len(pool_docs)
    base_record['pool_had_golden']      = pool_had_golden
    base_record['golden_added_to_pool'] = golden_added

    logger.info(
        f'    pool: {len(pool_docs)} chunks '
        f'(had_golden={pool_had_golden}, added={golden_added})'
    )

    # Rerank pool
    t0 = time.perf_counter()
    docs_reranked = chunker.rerank(question, pool_docs, top_k=RERANK_TOP_K)
    try:
        pairs  = [(question, d.page_content) for d in docs_reranked]
        scores = chunker.reranker.predict(pairs)
    except Exception:
        scores = [0.0] * len(docs_reranked)
    timings['reranking'] = round((time.perf_counter() - t0) * 1000, 1)

    reranked_isolated = [
        {
            'chunk_id':            d.metadata.get('chunk_id', '?'),
            'content':             d.page_content,
            'cross_encoder_score': float(s),
            'rank':                i + 1,
        }
        for i, (d, s) in enumerate(zip(docs_reranked, scores))
    ]

    top_ids = [r['chunk_id'] for r in reranked_isolated]
    logger.info(f'    rerank ({timings["reranking"]}ms): top={top_ids}')

    # RAGAs context_precision isolated
    # pass answer='' vì ContextPrecision dùng ground_truth (reference), không dùng answer
    t0 = time.perf_counter()
    post_rerank_contents = [c['content'] for c in reranked_isolated]
    ragas_result = ragas_wrapper.evaluate_reranker_and_generator(
        question=question,
        post_rerank_contexts=post_rerank_contents,
        answer='',              # ContextPrecision không dùng answer
        ground_truth=golden_answer,
        llm=judge_llm,
        embeddings=embeddings,
    )
    timings['ragas'] = round((time.perf_counter() - t0) * 1000, 1)

    ctx_pre = ragas_result.get('context_precision')
    _f = lambda v: f'{v:.2f}' if isinstance(v, float) else 'N/A'
    logger.info(f'    ragas ({timings["ragas"]}ms): ctx_pre_iso={_f(ctx_pre)}')

    return {
        **base_record,
        'reranked_isolated': reranked_isolated,
        'ragas_reranker_isolated': {'context_precision': ctx_pre},
        'timings_ms': timings,
    }


def run(collected_path: str = None, test_ids: list = None):
    logger.info('━' * 60)
    logger.info('[reranker] Phase 2 — Component Isolated: Reranker')
    logger.info('━' * 60)

    # Load data
    qa_pairs   = json.load(open(QA_PAIRS_PATH, encoding='utf-8'))
    chunks_raw = json.load(open(CHUNKS_DUMP_PATH, encoding='utf-8'))
    chunks_map = {c['chunk_id']: c for c in chunks_raw}

    if collected_path is None:
        collected_path = find_latest_collected(RESULTS_DIR)
    collected_data = json.load(open(collected_path, encoding='utf-8'))
    coll_map = {r['test_id']: r for r in collected_data['results']}

    if test_ids:
        qa_pairs = [qa for qa in qa_pairs if qa['id'] in test_ids]

    logger.info(f'[reranker] Test cases    : {len(qa_pairs)}')
    logger.info(f'[reranker] Corpus chunks : {len(chunks_map)}')
    logger.info(f'[reranker] Based on      : {os.path.basename(collected_path)}')

    judge_llm = build_judge_llm()
    chunker   = HierarchicalChunk()

    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'reranker_collected_{run_id}.json')

    results = []
    for i, qa in enumerate(qa_pairs, 1):
        group    = qa.get('group', '')
        q_short  = qa['question'][:72] + ('...' if len(qa['question']) > 72 else '')
        coll_res = coll_map.get(qa['id'], {})

        logger.info('')
        logger.info(f'  ┌─ [{i:>3}/{len(qa_pairs)}] {qa["id"]}  [{group}]')
        logger.info(f'  │  Q: {q_short}')
        logger.info('  │')

        current_judge = llm if group.startswith('D') else judge_llm

        try:
            record = process_one(qa, coll_res, chunks_map, chunker, current_judge)
            results.append(record)
            ctx_pre = record['ragas_reranker_isolated'].get('context_precision')
            _f = lambda v: f'{v:.2f}' if isinstance(v, float) else 'N/A'
            logger.info(
                f'  └─ ✓ ctx_pre_iso={_f(ctx_pre)} '
                f'| rerank={record["timings_ms"].get("reranking",0)}ms '
                f'ragas={record["timings_ms"].get("ragas",0)}ms'
            )
        except Exception as e:
            logger.error(f'  └─ ✗ FAILED: {e}', exc_info=True)
            results.append({'test_id': qa['id'], 'group': group, 'error': str(e)})

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'run_id':              run_id,
                'mode':                'reranker_isolated',
                'based_on_collected':  os.path.basename(collected_path),
                'config':              {'judge_model': JUDGE_MODEL or 'same_as_generator',
                                        'rerank_top_k': RERANK_TOP_K},
                'results':             results,
            }, f, ensure_ascii=False, indent=2)

    ok  = sum(1 for r in results if 'error' not in r)
    err = len(results) - ok
    logger.info('')
    logger.info('━' * 60)
    logger.info(f'[reranker] Done! ✓ {ok} / {len(qa_pairs)} | ✗ {err} failed')
    logger.info(f'[reranker] Output: {out_path}')
    logger.info('[reranker] Next: python -m eval.phase3_compute.compute_reranker')
    logger.info('━' * 60)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collected', type=str, default=None,
                        help='Path to collected JSON (default: latest)')
    parser.add_argument('--test-ids', type=str, default=None)
    args = parser.parse_args()
    run(
        collected_path=args.collected,
        test_ids=args.test_ids.split(',') if args.test_ids else None,
    )
