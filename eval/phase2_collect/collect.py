"""
eval/phase2_collect/collect.py
================================
PHASE 2: Chạy toàn bộ RAG pipeline và capture mọi intermediate output.

INPUT  : eval/data/faiss_eval_index/  (FAISS index từ phase0a)
         eval/data/qa_pairs.json      (test cases đã label từ phase1)
OUTPUT : eval/results/collected_YYYYMMDD_HHMMSS.json

Chạy lệnh:
    python3 -m eval.phase2_collect.collect
    python3 -m eval.phase2_collect.collect --test-ids q001,q002   # chạy subset

LƯU Ý QUAN TRỌNG:
    - Phase này là lần DUY NHẤT gọi LLM và RAGAs → tốn API cost
    - Mọi kết quả được lưu vào JSON → Phase 3 chỉ đọc lại, không gọi thêm API
    - Estimate: ~30-60 phút cho 17 test cases
    - Nếu crash giữa chừng: kết quả đã chạy vẫn được lưu vào file partial

LLM CALLS per test case:
    1. analyze_query          (1 LLM call)
    2. generate_answer        (1 LLM call)
    3. hallucination_check    (1 LLM call — nên dùng model KHÁC để giảm bias)
    4. RAGAs context_recall   (nhiều LLM calls nội bộ)
    5. RAGAs context_precision, faithfulness, answer_relevancy (nhiều LLM calls nội bộ)

CẤU HÌNH JUDGE LLM (để giảm self-judge bias):
    Thêm vào .env:
        GEMINI_API_KEY=your_gemini_api_key
        JUDGE_MODEL=gemini-1.5-flash
    Nếu không có → fallback dùng cùng LLM với generator (production behavior)
"""
import sys, os
import argparse
import json
from json_repair import repair_json
import time
import logging
from datetime import datetime

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

# ── App imports (cần .env hợp lệ) ──────────────────────────────────────────
from app.model import embeddings, llm
from app.service.LLM_service import llm_service
from app.service.RAG_services.RAG_service import RAGService
from app.service.RAG_services.ChunkSplitters.HierarchicalChunk import HierarchicalChunk
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import pickle

# ── Eval imports ─────────────────────────────────────────────────────────────
from eval.config import (
    FAISS_INDEX_DIR, QA_PAIRS_PATH, RESULTS_DIR,
    RETRIEVAL_TOP_K, RERANK_TOP_K,
    JUDGE_MODEL,
)
from eval.utils.token_counter import count_tokens
from eval.phase3_compute import ragas_wrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Tat log nhieu tu Gemini / Google GenAI SDK ────────────────────────────
import warnings
# Tat UserWarning tu langchain_google_genai (temperature ignored, v.v.)
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_google_genai')
# Tat AFC warnings tu google-genai SDK
warnings.filterwarnings('ignore', message='.*AFC.*')
warnings.filterwarnings('ignore', message='.*automatic function calling.*')
# Tat INFO/WARNING logger nhieu tu google generativeai internals
for _noisy_logger in [
    'google.ai.generativelanguage_v1beta',
    'google.generativeai',
    'google.api_core',
    'google.auth',
    'httpx',           # tat 'HTTP Request: POST ...' spam
    'httpcore',
]:
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

# Tat 'Calling LLM' va 'LLM generated answer' spam tu llm_service
logging.getLogger('app.service.LLM_service').setLevel(logging.WARNING)


# ── Judge LLM ─────────────────────────────────────────────────────────────────
def build_judge_llm():
    """
    Tạo judge LLM riêng để đánh giá hallucination.
    Dùng JUDGE_MODEL qua OpenRouter nếu được cấu hình trong .env.
    Nếu không → dùng cùng llm với generator (production behavior, có self-bias).

    Cách cấu hình:
        .env:
            JUDGE_MODEL=google/gemini-flash-1.5   # hoac bat ky model OpenRouter nao
            # GEMINI_API_KEY khong can thiet khi dung qua OpenRouter
    """
    if JUDGE_MODEL:
        from langchain_openai import ChatOpenAI
        from app.core.env_config import settings
        logger.info(f'[judge] Dung judge model qua OpenRouter: {JUDGE_MODEL}')
        return ChatOpenAI(
            model=JUDGE_MODEL,
            api_key=settings.api_key,
            base_url='https://openrouter.ai/api/v1',
            temperature=0.0,
            max_tokens=16384,
            request_timeout=120,
        )
    logger.info('[judge] Dung cung LLM voi generator (self-judge, co self-bias)')
    return llm


def extract_model_name(response) -> str:
    """
    Extract actual model name từ LangChain AIMessage.response_metadata.
    OpenRouter trả về tên model thực trong metadata (khác với configured name 'openrouter/free').
    """
    meta = getattr(response, 'response_metadata', {}) or {}
    return (
        meta.get('model_name') or
        meta.get('model') or
        getattr(response, 'model_name', None) or
        'unknown'
    )


# ── Setup một lần ─────────────────────────────────────────────────────────────
def setup_retrievers():
    """
    Load FAISS và build BM25 từ eval index.
    Chạy 1 lần trước vòng lặp test cases.
    """
    if not os.path.exists(os.path.join(FAISS_INDEX_DIR, 'index.faiss')):
        raise FileNotFoundError(
            f'\n[ERROR] Không tìm thấy FAISS index tại: {FAISS_INDEX_DIR}\n'
            f'Chạy phase0a trước: python3 -m eval.phase0_setup.embed_and_index'
        )

    logger.info('[setup] Loading FAISS index...')
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVAL_TOP_K})

    logger.info('[setup] Building BM25 retriever...')
    with open(os.path.join(FAISS_INDEX_DIR, 'index.pkl'), 'rb') as f:
        raw = pickle.load(f)
    all_docs = list(raw[0]._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=RETRIEVAL_TOP_K)

    logger.info(f'[setup] Ready. {len(all_docs)} chunks loaded.')
    return faiss_retriever, bm25_retriever


def make_chunk_list(docs, score_key: str = '_rrf_score') -> list:
    """
    Convert List[Document] → List[dict] serializable.
    Gán chunk_id từ chunk_index metadata.
    """
    return [
        {
            'chunk_id': f"chunk_{d.metadata.get('chunk_index', 0):04d}",
            'content':  d.page_content,
            score_key:  float(d.metadata.get(score_key, 0.0)),
        }
        for d in docs
    ]


# ── RAGService wrapper (tránh gọi load_global_index trong __init__) ──────────
class _MinimalRAGService:
    """
    Subset của RAGService chỉ chứa những method cần cho eval:
      - multi_query_hybrid_search
      - relevance_and_sufficiency_check
      - build_block_context

    Không gọi load_global_index() (không cần government_data/ cho eval).
    """
    from app.core.env_config import settings as _settings

    def _rrf_score(self, bm25_results, faiss_results, bm25_w=None, faiss_w=None, k=60):
        """Copy từ RAGService._rrf_score."""
        from app.core.env_config import settings
        bm25_w = bm25_w or settings.bm25_w
        faiss_w = faiss_w or settings.faiss_w
        scores = {}
        all_docs = {}
        for rank, doc in enumerate(bm25_results):
            content = doc.page_content
            scores[content] = scores.get(content, 0) + bm25_w / (rank + k)
            all_docs[content] = doc
        for rank, doc in enumerate(faiss_results):
            content = doc.page_content
            scores[content] = scores.get(content, 0) + faiss_w / (rank + k)
            all_docs[content] = doc
        return scores, all_docs

    def multi_query_hybrid_search(
        self, question, bm25_retriever, faiss_retriever,
        generated_queries, filter_dict=None, top_k=None
    ):
        """Hybrid BM25 + FAISS + RRF merge. Copy logic từ RAGService."""
        from app.core.env_config import settings
        top_k = top_k or settings.top_k
        accumulated_scores = {}
        accumulated_docs = {}

        for query in generated_queries:
            bm25_results  = bm25_retriever.invoke(query)
            faiss_results = faiss_retriever.invoke(query)

            if filter_dict:
                def is_valid(d):
                    for k, v in filter_dict.items():
                        doc_val = d.metadata.get(k)
                        if doc_val is not None and doc_val != v:
                            return False
                    return True
                bm25_results  = [d for d in bm25_results if is_valid(d)]
                faiss_results = [d for d in faiss_results if is_valid(d)]

            chunk_scores, docs = self._rrf_score(bm25_results, faiss_results)
            for content, score in chunk_scores.items():
                accumulated_scores[content] = accumulated_scores.get(content, 0) + score
            accumulated_docs.update(docs)

        ranked = sorted(accumulated_scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for content, score in ranked[:top_k]:
            doc = accumulated_docs[content]
            doc.metadata['_rrf_score'] = round(score, 6)
            result.append(doc)
        return result

    def relevance_and_sufficiency_check(self, docs):
        """Kiểm tra docs có relevant và sufficient không. Copy từ RAGService."""
        from app.core.env_config import settings
        if not docs:
            return False, False
        is_relevant  = len(docs) >= settings.min_relevant_docs
        max_score    = max(d.metadata.get('_rrf_score', 0) for d in docs)
        is_sufficient = max_score >= settings.relevance_threshold
        return is_relevant, is_sufficient

    def build_block_context(self, docs):
        """Format context cho generator. Copy từ RAGService."""
        blocks = []
        for i, doc in enumerate(docs, start=1):
            page      = doc.metadata.get('page')
            file_name = doc.metadata.get('file_name', 'Tài liệu đánh giá')
            page_info = f' | Trang {page}' if page is not None else ''
            blocks.append(
                f'[{i}] Nguồn: {file_name}{page_info}\n'
                f'Trích xuất: {doc.page_content}'
            )
        return '\n\n'.join(blocks)


# ── Core: process 1 test case ───────────────────────────────────────────────────────
def process_one(qa: dict, faiss_retriever, bm25_retriever, chunker, rag, judge_llm) -> dict:
    """
    Chạy toàn bộ pipeline cho 1 test case và capture tất cả outputs.

    INPUT : qa        = một test case từ qa_pairs.json
            judge_llm = LLM dùng cho hallucination_check trong test case này:
                        - Group D* → llm (cùng model với generator, để đo self-bias)
                        - Còn lại → judge_llm (Gemini hoặc model mạnh hơn, factual consistency chính xác)
    OUTPUT: dict với đầy đủ step_outputs, timings, tokens, models_used
    """
    question     = qa['question']
    golden_answer = qa.get('golden_answer', '')

    models_used = {}
    tokens      = {}
    timings     = {}

    # ── Bước 1: analyze_query ────────────────────────────────────────────────
    logger.info('  ├─[1/8] analyze_query')
    t0 = time.perf_counter()
    analyze_content = llm_service.format_user_content('analyze_query', question=question)
    analyze_raw     = llm_service.ask_model(llm, 'analyze_query', analyze_content)
    timings['analyze_query'] = round((time.perf_counter() - t0) * 1000, 1)

    models_used['analyze_query'] = extract_model_name(analyze_raw)
    tokens['analyze_in']  = count_tokens(analyze_content)
    tokens['analyze_out'] = count_tokens(analyze_raw.content)

    try:
        analysis = json.loads(analyze_raw.content.strip())
        if analysis.get('type') not in ('chit_chat', 'simple', 'complex'):
            raise ValueError('invalid type')
    except Exception:
        analysis = {'type': 'simple', 'round1': [question], 'round2': []}

    logger.info(f'  │       type={analysis["type"]}  queries={len(analysis.get("round1", []))}  ({timings["analyze_query"]}ms)')

    # ── Bước 2: Retrieval Round 1 ────────────────────────────────────────────
    logger.info('  ├─[2/8] hybrid retrieval')
    t0 = time.perf_counter()
    docs_r1 = rag.multi_query_hybrid_search(
        question, bm25_retriever, faiss_retriever,
        generated_queries=analysis.get('round1', [question]),
        filter_dict=analysis.get('filter', {}),
    )
    timings['retrieval_r1'] = round((time.perf_counter() - t0) * 1000, 1)
    retrieval_r1 = make_chunk_list(docs_r1, '_rrf_score')
    logger.info(f'  │       retrieved {len(retrieval_r1)} chunks  ({timings["retrieval_r1"]}ms)')

    # ── Bước 3: Rerank ───────────────────────────────────────────────────────
    logger.info('  ├─[3/8] rerank')
    t0 = time.perf_counter()
    docs_reranked = chunker.rerank(question, docs_r1, top_k=RERANK_TOP_K)
    timings['reranking'] = round((time.perf_counter() - t0) * 1000, 1)

    # chunker.rerank() chỉ trả về List[Document], không trả về scores.
    # Chạy lại CrossEncoder để lấy scores cho logging.
    try:
        pairs  = [(question, d.page_content) for d in docs_reranked]
        scores = chunker.reranker.predict(pairs)
        reranking = [
            {
                'chunk_id':            f"chunk_{d.metadata.get('chunk_index', 0):04d}",
                'content':             d.page_content,
                'cross_encoder_score': float(s),
            }
            for d, s in zip(docs_reranked, scores)
        ]
    except Exception:
        # Fallback nếu không lấy được scores
        reranking = [
            {
                'chunk_id': f"chunk_{d.metadata.get('chunk_index', 0):04d}",
                'content':  d.page_content,
                'cross_encoder_score': 0.0,
            }
            for d in docs_reranked
        ]
    logger.info(f'  │       top-{len(reranking)} chunks kept  ({timings["reranking"]}ms)')

    # ── Bước 4: Relevance check & Retry ─────────────────────────────────────
    logger.info('  ├─[4/8] relevance check')
    is_relevant, is_sufficient = rag.relevance_and_sufficiency_check(docs_reranked)
    retrieval_r2 = None
    timings['retrieval_r2'] = None
    used_round = 'round1'
    final_retrieved = retrieval_r1

    if not (is_relevant and is_sufficient) and analysis.get('round2'):
        logger.info('  │       → not sufficient, retrying with round2')
        t0 = time.perf_counter()
        docs_r2 = rag.multi_query_hybrid_search(
            question, bm25_retriever, faiss_retriever,
            generated_queries=analysis['round2'],
            filter_dict=analysis.get('filter', {}),
        )
        timings['retrieval_r2'] = round((time.perf_counter() - t0) * 1000, 1)
        retrieval_r2 = make_chunk_list(docs_r2, '_rrf_score')

        docs_reranked = chunker.rerank(question, docs_r2, top_k=RERANK_TOP_K)
        try:
            pairs  = [(question, d.page_content) for d in docs_reranked]
            scores = chunker.reranker.predict(pairs)
            reranking = [
                {
                    'chunk_id':            f"chunk_{d.metadata.get('chunk_index', 0):04d}",
                    'content':             d.page_content,
                    'cross_encoder_score': float(s),
                }
                for d, s in zip(docs_reranked, scores)
            ]
        except Exception:
            reranking = [
                {'chunk_id': f"chunk_{d.metadata.get('chunk_index', 0):04d}",
                 'content': d.page_content, 'cross_encoder_score': 0.0}
                for d in docs_reranked
            ]
        used_round    = 'round2'
        final_retrieved = retrieval_r2

    # ── Bước 5: Build context ────────────────────────────────────────────────
    context = rag.build_block_context(docs_reranked)

    # ── Bước 6: Generate answer ──────────────────────────────────────────────
    logger.info('  ├─[6/8] generate answer')
    t0 = time.perf_counter()
    gen_content = llm_service.format_user_content(
        'question_answer', context=context, question=question
    )
    gen_raw = llm_service.ask_model(llm, 'question_answer', gen_content, [])
    timings['generation'] = round((time.perf_counter() - t0) * 1000, 1)

    models_used['generation'] = extract_model_name(gen_raw)
    tokens['gen_in']  = count_tokens(gen_content)
    tokens['gen_out'] = count_tokens(gen_raw.content)
    generated_answer  = gen_raw.content

    # ── Bước 7: Hallucination check (judge LLM) ──────────────────────────────
    logger.info('  ├─[7/8] hallucination check')
    t0 = time.perf_counter()
    hcheck_content = llm_service.format_user_content(
        'hallucination_check',
        context=context, question=question, answer=generated_answer
    )
    hcheck_raw = llm_service.ask_model(judge_llm, 'hallucination_check', hcheck_content)
    timings['hallucination_check'] = round((time.perf_counter() - t0) * 1000, 1)

    models_used['hallucination_check'] = extract_model_name(hcheck_raw)
    tokens['hcheck_in']  = count_tokens(hcheck_content)
    tokens['hcheck_out'] = count_tokens(hcheck_raw.content)

    # Normalize content: Gemini tra ve list parts, OpenAI/OpenRouter tra ve str
    _hcheck_text = hcheck_raw.content
    if isinstance(_hcheck_text, list):
        _hcheck_text = ' '.join(
            p.get('text', '') if isinstance(p, dict) else str(p)
            for p in _hcheck_text
        )

    _hcheck_parse_error = False
    try:
        hcheck_result = json.loads(_hcheck_text.strip())
    except Exception:
        try:
            repaired = repair_json(_hcheck_text.strip(), return_objects=True)
            hcheck_result = repaired if isinstance(repaired, dict) else {}
        except Exception:
            hcheck_result = {}
    # Nếu sau cả 2 bước vẫn không parse được confidence → ghi lại raw text
    if 'confidence' not in hcheck_result:
        _hcheck_parse_error = True
        logger.warning(f'  └─[hcheck] JSON parse failed. Raw response: {_hcheck_text[:200]!r}')
    hcheck_result = {
        'confidence':        float(hcheck_result.get('confidence', 1.0)),
        'reason':            str(hcheck_result.get('reason', _hcheck_text.strip()[:500]))
                             if _hcheck_parse_error else
                             str(hcheck_result.get('reason', '')),
        'hcheck_parse_error': _hcheck_parse_error,
    }

    # Flag nếu model swap giữa các calls trong cùng test case
    model_names = [v for k, v in models_used.items()]
    models_used['model_consistent'] = len(set(model_names)) == 1

    # ── Bước 8: RAGAs ────────────────────────────────────────────────────────
    logger.info('  ├─[8/8] RAGAs evaluation')
    t0 = time.perf_counter()

    # Call 1: Retriever — context_recall dùng pre-rerank chunks
    pre_rerank_contents = [c['content'] for c in final_retrieved]
    ragas_retriever = ragas_wrapper.evaluate_retriever(
        question=question,
        pre_rerank_contexts=pre_rerank_contents,
        ground_truth=golden_answer,
        llm=judge_llm,
        embeddings=embeddings,
    )

    # Call 2: Reranker + Generator — context_precision, faithfulness, answer_relevancy
    post_rerank_contents = [c['content'] for c in reranking]
    ragas_gen = ragas_wrapper.evaluate_reranker_and_generator(
        question=question,
        post_rerank_contexts=post_rerank_contents,
        answer=generated_answer,
        ground_truth=golden_answer,
        llm=judge_llm,
        embeddings=embeddings,
    )

    timings['ragas'] = round((time.perf_counter() - t0) * 1000, 1)
    ragas_scores = {**ragas_retriever, **ragas_gen}

    # ── E2E Latency (không tính RAGAs) ────────────────────────────────────────
    timings['total_e2e'] = round(
        timings['analyze_query'] +
        timings['retrieval_r1'] +
        timings['reranking'] +
        (timings['retrieval_r2'] or 0) +
        timings['generation'] +
        timings['hallucination_check'],
        1
    )

    # ── Tổng token ─────────────────────────────────────────────────────────────
    tokens['total'] = sum(
        v for k, v in tokens.items() if isinstance(v, (int, float))
    )

    # ── Build record ─────────────────────────────────────────────────────────
    return {
        'test_id':          qa['id'],
        'group':            qa.get('group', 'unknown'),
        'question':         question,
        'golden_answer':    golden_answer,
        'golden_chunk_ids': qa.get('golden_chunk_ids', []),

        'models_used':      models_used,

        'step_outputs': {
            'analyze_query':             analysis,
            'retrieval_round1':          retrieval_r1,
            'retrieval_round2':          retrieval_r2,
            'retrieval_final':           final_retrieved,
            'used_round':                used_round,
            'reranking':                 reranking,
            'context_sent_to_generator': context,
            'generated_answer':          generated_answer,
            'hallucination_check':       hcheck_result,
            'ragas':                     ragas_scores,
        },

        'timings_ms':       timings,
        'tokens_estimated': tokens,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run(test_ids: list = None):
    """
    Chạy toàn bộ eval collect phase.

    INPUT : test_ids = None → chạy tất cả 17 cases
                     = ['q001', 'q002'] → chạy subset
    OUTPUT: eval/results/collected_YYYYMMDD_HHMMSS.json
    """
    # Load qa_pairs
    if not os.path.exists(QA_PAIRS_PATH):
        raise FileNotFoundError(
            f'\n[ERROR] Không tìm thấy: {QA_PAIRS_PATH}\n'
            f'Tạo file này trước (phase1 labeling).'
        )
    with open(QA_PAIRS_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    # Filter theo test_ids nếu có
    if test_ids:
        qa_pairs = [qa for qa in qa_pairs if qa['id'] in test_ids]
        logger.info(f'[collect] Running subset: {[qa["id"] for qa in qa_pairs]}')
    logger.info(f'[collect] Total test cases: {len(qa_pairs)}')

    # Setup
    faiss_retriever, bm25_retriever = setup_retrievers()
    chunker    = HierarchicalChunk()
    rag        = _MinimalRAGService()
    judge_llm  = build_judge_llm()

    from app.core.env_config import settings

    run_id    = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(RESULTS_DIR, f'collected_{run_id}.json')

    # Chạy từng test case
    results = []
    for i, qa in enumerate(qa_pairs, 1):
        group = qa.get('group', '')
        q_short = qa['question'][:72] + ('...' if len(qa['question']) > 72 else '')
        logger.info('')
        logger.info(f'  ┌─ [{i:>2}/{len(qa_pairs)}] {qa["id"]}  [{group}]')
        logger.info(f'  │  Q: {q_short}')

        # Chọn judge LLM theo group:
        #   D* (bias cases) → SAME model với generator (cố ý để đo self-bias)
        #   Còn lại    → judge_llm (Gemini, factual consistency chính xác)
        if group.startswith('D'):
            current_judge = llm
            logger.info('  │  judge: same-as-generator (self-bias group)')
        else:
            current_judge = judge_llm
            judge_name = JUDGE_MODEL or 'same_as_generator'
            logger.info(f'  │  judge: {judge_name}')
        logger.info('  │')

        try:
            record = process_one(qa, faiss_retriever, bm25_retriever, chunker, rag, current_judge)
            results.append(record)
            e2e   = record['timings_ms']['total_e2e']
            toks  = record['tokens_estimated']['total']
            conf  = record['step_outputs']['hallucination_check'].get('confidence', '?')
            ragas = record['step_outputs']['ragas']
            cr  = ragas.get('context_recall')
            cp  = ragas.get('context_precision')
            ff  = ragas.get('faithfulness')
            ar  = ragas.get('answer_relevancy')
            _f  = lambda v: f'{v:.2f}' if isinstance(v, float) else 'N/A'
            logger.info('  │')
            logger.info(f'  │  hallucination confidence : {conf}')
            logger.info(f'  │  context_recall           : {_f(cr)}')
            logger.info(f'  │  context_precision        : {_f(cp)}')
            logger.info(f'  │  faithfulness             : {_f(ff)}')
            logger.info(f'  │  answer_relevancy         : {_f(ar)}')
            logger.info(f'  └─ ✓ e2e={e2e}ms  tokens={toks}')
        except Exception as e:
            logger.error(f'  └─ ✗ FAILED: {e}', exc_info=True)
            # Lưu error record để không mất thứ tự
            results.append({
                'test_id': qa['id'],
                'group':   qa.get('group', 'unknown'),
                'error':   str(e),
            })

        # Auto-save sau mỗi test case (partial save)
        output = {
            'run_id': run_id,
            'config': {
                'strategy':      settings.strategy,
                'top_k':         settings.top_k,
                'rerank_top_k':  RERANK_TOP_K,
                'temperature':   settings.temperature,
                'judge_model':   JUDGE_MODEL or 'same_as_generator',
            },
            'results': results,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    ok  = sum(1 for r in results if 'error' not in r)
    err = len(results) - ok
    logger.info('')
    logger.info('━' * 60)
    logger.info(f'[collect] Hoàn thành!  ✓ {ok} passed  ✗ {err} failed  / {len(qa_pairs)} total')
    logger.info(f'[collect] Output: {out_path}')
    logger.info('[collect] Next: python3 -m eval.phase3_compute.compute_metrics')
    logger.info('━' * 60)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Phase 2: Collect')
    parser.add_argument(
        '--test-ids', type=str, default=None,
        help='Comma-separated list of test IDs to run, e.g. q001,q002'
    )
    args = parser.parse_args()
    test_ids = args.test_ids.split(',') if args.test_ids else None
    run(test_ids=test_ids)
