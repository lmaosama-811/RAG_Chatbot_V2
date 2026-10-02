"""
eval/phase2_collect/generator.py
=================================
PHASE 2 (Component Isolated): Generator Eval

Chạy Generator ở điều kiện lý tưởng:
    question + golden_context (từ golden_chunk_ids) → generated_answer_isolated
    → RAGAs: faithfulness_iso, answer_relevancy_iso

Hoàn toàn độc lập với collect.py. Chỉ cần:
    - eval/data/qa_pairs.json
    - eval/data/chunks_dump.json
    - App LLM (llm, embeddings từ app.model)

Output: eval/results/generator_collected_<YYYYMMDD_HHMMSS>.json

Chạy:
    python -m eval.phase2_collect.generator
    python -m eval.phase2_collect.generator --test-ids q001,q002
"""
import sys, os
import argparse
import json
import time
import logging
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from app.model import embeddings, llm
from app.service.LLM_service import llm_service

from eval.config import QA_PAIRS_PATH, CHUNKS_DUMP_PATH, RESULTS_DIR
from eval.phase3_compute import ragas_wrapper
from eval.phase2_collect.collect import build_judge_llm, extract_model_name, JUDGE_MODEL

for _n in ['google.ai.generativelanguage_v1beta', 'google.generativeai',
           'httpx', 'httpcore', 'langchain_core']:
    logging.getLogger(_n).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def build_golden_context(golden_chunk_ids: list, chunks_map: dict):
    """Format golden chunks thành context string (cùng format với RAGService)."""
    found_all = True
    parts = []
    for i, cid in enumerate(golden_chunk_ids, 1):
        chunk = chunks_map.get(cid)
        if chunk is None:
            logger.warning(f'    [!] golden chunk {cid!r} không tìm thấy trong chunks_dump')
            found_all = False
            continue
        parts.append(
            f'[{i}] Nguồn: Tài liệu đánh giá\n'
            f'Trích xuất: {chunk["content"]}'
        )
    if not parts:
        return None, False
    return '\n\n'.join(parts), found_all


def process_one(qa: dict, chunks_map: dict, judge_llm) -> dict:
    """
    Isolated Generator eval cho 1 test case.
    Flow: golden context → generate → RAGAs (faith + ansrel).
    """
    test_id          = qa['id']
    group            = qa.get('group', '')
    question         = qa['question']
    golden_answer    = qa.get('golden_answer', '')
    golden_chunk_ids = qa.get('golden_chunk_ids', [])

    timings     = {}
    base_record = {
        'test_id':                   test_id,
        'group':                     group,
        'question':                  question,
        'golden_answer':             golden_answer,
        'golden_chunk_ids':          golden_chunk_ids,
        'golden_context':            None,
        'golden_chunks_found':       False,
        'generated_answer_isolated': None,
        'ragas_isolated': {'faithfulness': None, 'answer_relevancy': None},
        'timings_ms':                {},
        'model_used':                None,
    }

    if not golden_chunk_ids:
        logger.warning(f'    [!] {test_id}: không có golden_chunk_ids → skip')
        return base_record

    golden_context, found_all = build_golden_context(golden_chunk_ids, chunks_map)
    if golden_context is None:
        logger.warning(f'    [!] {test_id}: không tìm được golden chunks → skip')
        return base_record

    base_record['golden_context']      = golden_context
    base_record['golden_chunks_found'] = found_all

    # Generate với golden context
    t0 = time.perf_counter()
    gen_content = llm_service.format_user_content(
        'question_answer', context=golden_context, question=question
    )
    gen_raw = llm_service.ask_model(llm, 'question_answer', gen_content, [])
    timings['generation'] = round((time.perf_counter() - t0) * 1000, 1)

    generated_answer_isolated = gen_raw.content
    try:
        model_used = extract_model_name(gen_raw)
    except Exception:
        model_used = ''

    logger.info(f'    gen ({timings["generation"]}ms): {generated_answer_isolated[:80]}...')

    # RAGAs với golden context
    t0 = time.perf_counter()
    golden_chunk_contents = [
        chunks_map[cid]['content'] for cid in golden_chunk_ids if cid in chunks_map
    ]
    ragas_result = ragas_wrapper.evaluate_reranker_and_generator(
        question=question,
        post_rerank_contexts=golden_chunk_contents,
        answer=generated_answer_isolated,
        ground_truth=golden_answer,
        llm=judge_llm,
        embeddings=embeddings,
    )
    timings['ragas'] = round((time.perf_counter() - t0) * 1000, 1)

    _f = lambda v: f'{v:.2f}' if isinstance(v, float) else 'N/A'
    logger.info(
        f'    ragas ({timings["ragas"]}ms): '
        f'faith_iso={_f(ragas_result.get("faithfulness"))} '
        f'ansrel_iso={_f(ragas_result.get("answer_relevancy"))}'
    )

    return {
        **base_record,
        'generated_answer_isolated': generated_answer_isolated,
        'ragas_isolated': {
            'faithfulness':     ragas_result.get('faithfulness'),
            'answer_relevancy': ragas_result.get('answer_relevancy'),
        },
        'timings_ms': timings,
        'model_used': model_used,
    }


def run(test_ids: list = None):
    logger.info('━' * 60)
    logger.info('[generator] Phase 2 — Component Isolated: Generator')
    logger.info('━' * 60)

    qa_pairs   = json.load(open(QA_PAIRS_PATH, encoding='utf-8'))
    chunks_raw = json.load(open(CHUNKS_DUMP_PATH, encoding='utf-8'))
    chunks_map = {c['chunk_id']: c for c in chunks_raw}

    if test_ids:
        qa_pairs = [qa for qa in qa_pairs if qa['id'] in test_ids]
    logger.info(f'[generator] Test cases: {len(qa_pairs)} | Corpus: {len(chunks_map)} chunks')

    judge_llm = build_judge_llm()
    run_id    = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(RESULTS_DIR, f'generator_collected_{run_id}.json')

    results = []
    for i, qa in enumerate(qa_pairs, 1):
        group   = qa.get('group', '')
        q_short = qa['question'][:72] + ('...' if len(qa['question']) > 72 else '')
        logger.info('')
        logger.info(f'  ┌─ [{i:>3}/{len(qa_pairs)}] {qa["id"]}  [{group}]')
        logger.info(f'  │  Q: {q_short}')
        logger.info('  │')

        current_judge = llm if group.startswith('D') else judge_llm

        try:
            record = process_one(qa, chunks_map, current_judge)
            results.append(record)
            _f = lambda v: f'{v:.2f}' if isinstance(v, float) else 'N/A'
            faith  = record['ragas_isolated'].get('faithfulness')
            ansrel = record['ragas_isolated'].get('answer_relevancy')
            logger.info(
                f'  └─ ✓ faith_iso={_f(faith)}  ansrel_iso={_f(ansrel)} '
                f'| gen={record["timings_ms"].get("generation",0)}ms '
                f'ragas={record["timings_ms"].get("ragas",0)}ms'
            )
        except Exception as e:
            logger.error(f'  └─ ✗ FAILED: {e}', exc_info=True)
            results.append({'test_id': qa['id'], 'group': group, 'error': str(e)})

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'run_id': run_id, 'mode': 'generator_isolated',
                'config': {'judge_model': JUDGE_MODEL or 'same_as_generator'},
                'results': results,
            }, f, ensure_ascii=False, indent=2)

    ok  = sum(1 for r in results if 'error' not in r)
    err = len(results) - ok
    logger.info('')
    logger.info('━' * 60)
    logger.info(f'[generator] Done! ✓ {ok} / {len(qa_pairs)} | ✗ {err} failed')
    logger.info(f'[generator] Output: {out_path}')
    logger.info('[generator] Next: python -m eval.phase3_compute.compute_generator')
    logger.info('━' * 60)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-ids', type=str, default=None)
    args = parser.parse_args()
    run(test_ids=args.test_ids.split(',') if args.test_ids else None)
