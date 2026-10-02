"""
eval/utils/metrics.py
=====================
Helpers cho ranx và Semantic Similarity.

ranx được dùng cho:
  - Hit Rate@5 (retriever): có ít nhất 1 golden chunk trong top-5 không?
  - MRR (reranker): rank của golden chunk đầu tiên sau rerank là bao nhiêu?

SentenceTransformer dùng cho Semantic Similarity (Tier 2), chạy local.

QUAN TRỌNG về MRR:
  cross_encoder_score từ CrossEncoder có thể âm.
  ranx yêu cầu positive scores để sort đúng.
  Hàm build_run_from_stage() tự động shift scores về dương.
"""
import numpy as np
from ranx import Qrels, Run


class SemSimilarityScorer:
    """
    Singleton: load sentence-transformer model 1 lần, tái sử dụng cho mọi lần gọi.
    Model: all-MiniLM-L6-v2 — nhỏ, nhanh, đủ tốt cho similarity tiếng Việt cơ bản.
    """
    _instance = None
    _model = None

    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            from sentence_transformers import SentenceTransformer
            print(f'[SemSim] Loading {model_name} ...')
            cls._model = SentenceTransformer(model_name)
            print('[SemSim] Ready.')
        return cls._instance

    def score(self, text1: str, text2: str) -> float:
        """Cosine similarity giữa 2 chuỗi text. Output: float trong [-1, 1]."""
        if not text1 or not text2:
            return 0.0
        emb = self._model.encode([text1, text2])
        return float(
            np.dot(emb[0], emb[1]) /
            (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-10)
        )

    def batch_score(self, pairs: list) -> list:
        """
        Tính similarity nhiều cặp cùng lúc — nhanh hơn gọi score() từng cái.

        INPUT : [(text1a, text1b), (text2a, text2b), ...]
        OUTPUT: [score1, score2, ...]
        """
        if not pairs:
            return []
        texts = [t for pair in pairs for t in pair]
        embs = self._model.encode(texts)
        return [
            float(
                np.dot(embs[i], embs[i + 1]) /
                (np.linalg.norm(embs[i]) * np.linalg.norm(embs[i + 1]) + 1e-10)
            )
            for i in range(0, len(embs), 2)
        ]


def build_qrels(qa_pairs: list) -> Qrels:
    """
    Tạo ranx.Qrels từ danh sách qa_pairs.

    INPUT : qa_pairs = [{"id": "q001", "golden_chunk_ids": ["chunk_0003", "chunk_0007"]}, ...]
    OUTPUT: ranx.Qrels object dùng cho evaluate()

    Relevance là binary (0 hoặc 1) vì chúng ta không có graded labels.

    Ví dụ:
        qrels = build_qrels(qa_pairs)
        results = evaluate(qrels, run, ["hit_rate@5", "mrr"])
    """
    data = {}
    for qa in qa_pairs:
        golden = qa.get('golden_chunk_ids', [])
        if golden:
            # relevance = 1 (binary)
            data[qa['id']] = {cid: 1 for cid in golden}
    return Qrels(data)


def build_run_from_stage(collected_results: list, stage: str = 'retrieval') -> Run:
    """
    Tạo ranx.Run từ collected_outputs.json cho một stage cụ thể.

    INPUT:
        collected_results : list kết quả từ phase2_collect/collect.py
        stage             : 'retrieval' (trước rerank) hoặc 'reranking' (sau rerank)

    OUTPUT: ranx.Run object

    Score mapping:
        retrieval → rrf_score (từ RRF merge, luôn dương)
        reranking → cross_encoder_score shifted về dương (cross-encoder có thể âm)

    Ví dụ:
        run_retrieval = build_run_from_stage(results, stage='retrieval')
        run_reranker  = build_run_from_stage(results, stage='reranking')

        hit_rate = evaluate(qrels, run_retrieval, ["hit_rate@5"])
        mrr      = evaluate(qrels, run_reranker,  ["mrr"])
    """
    data = {}
    for result in collected_results:
        test_id = result['test_id']

        if stage == 'retrieval':
            chunks = result.get('step_outputs', {}).get('retrieval_final', [])
            score_key = 'rrf_score'
        else:
            chunks = result.get('step_outputs', {}).get('reranking', [])
            score_key = 'cross_encoder_score'

        if not chunks:
            continue

        raw_scores = {c['chunk_id']: float(c.get(score_key, 0.0)) for c in chunks}

        # Shift scores nếu có giá trị âm (chỉ xảy ra với cross_encoder_score)
        if stage == 'reranking':
            min_score = min(raw_scores.values())
            if min_score < 0:
                raw_scores = {k: v - min_score + 1e-6 for k, v in raw_scores.items()}

        data[test_id] = raw_scores

    return Run(data)
