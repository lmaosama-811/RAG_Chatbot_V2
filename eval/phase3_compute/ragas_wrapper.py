"""
eval/phase3_compute/ragas_wrapper.py
======================================
Wrap RAGAs evaluate() API.

Được gọi từ phase2_collect/collect.py để tính RAGAs scores trong khi collect.
Kết quả được lưu vào collected_outputs.json — phase3 chỉ đọc lại, không tính lại.

RAGAs metrics và cách tính:

  context_recall (RETRIEVER):
    Input : question, contexts (pre-rerank), ground_truth
    Tính  : LLM hỏi "Mỗi câu trong ground_truth có được support bởi context không?"
    Score : # sentences supported / total sentences in ground_truth
    Cao   : Context cover đủ thông tin cần thiết
    Thấp  : Retriever bỏ sót chunk quan trọng

  context_precision (RERANKER):
    Input : question, contexts (post-rerank), ground_truth
    Tính  : LLM hỏi "Mỗi retrieved chunk có thực sự useful không?"
    Score : Weighted precision (vị trí chunk quan trọng: chunk đầu có weight cao hơn)
    Cao   : Reranker đẩy chunks relevant lên top, loại chunks nhiễu
    Thấp  : Reranker sắp xếp sai, chunks nhiễu lẫn vào top

  faithfulness (GENERATOR):
    Input : question, answer, contexts (post-rerank)
    Tính  : LLM decompose answer thành N claims, verify từng claim vs context
    Score : # claims supported / N claims
    Cao   : Answer grounded trong context
    Thấp  : Answer hallucinate (add info không có trong context)

  answer_relevancy (GENERATOR):
    Input : question, answer
    Tính  : LLM generate K synthetic questions từ answer, cosine similarity với original
    Score : mean(cosine_similarity(synthetic_question, original_question))
    Cao   : Answer trả lời đúng câu hỏi
    Thấp  : Answer off-topic / quá verbose / trả lời câu hỏi khác
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
logger = logging.getLogger(__name__)


def _build_ragas_llm(llm):
    """Wrap LangChain LLM thành RagasLLM."""
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings(embeddings):
    """Wrap LangChain Embeddings thành RagasEmbeddings."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    return LangchainEmbeddingsWrapper(embeddings)


def evaluate_retriever(
    question: str,
    pre_rerank_contexts: list,
    ground_truth: str,
    llm,
    embeddings,
) -> dict:
    """
    Tính context_recall cho Retriever stage.

    INPUT:
        question            : câu hỏi của test case
        pre_rerank_contexts : List[str] — contents của chunks TRƯỚC rerank
        ground_truth        : golden_answer từ qa_pairs.json
        llm                 : LangChain ChatModel instance
        embeddings          : LangChain Embeddings instance

    OUTPUT: {"context_recall": float}

    Nếu RAGAs lỗi (parse error, timeout, ...): trả về {"context_recall": None}
    """
    from ragas import evaluate
    from ragas.metrics import context_recall
    from datasets import Dataset

    try:
        dataset = Dataset.from_dict({
            'question':     [question],
            'contexts':     [pre_rerank_contexts],
            'ground_truth': [ground_truth],
            'answer':       [''],       # context_recall không cần answer
        })
        result = evaluate(
            dataset,
            metrics=[context_recall],
            llm=_build_ragas_llm(llm),
            embeddings=_build_ragas_embeddings(embeddings),
            raise_exceptions=True,
        )
        val = result['context_recall']
        if isinstance(val, list):
            val = val[0] if val else None
        return {'context_recall': float(val) if val is not None else None}
    except Exception as e:
        logger.warning(f'[RAGAs] context_recall failed: {e}')
        return {'context_recall': None}


def evaluate_reranker_and_generator(
    question: str,
    post_rerank_contexts: list,
    answer: str,
    ground_truth: str,
    llm,
    embeddings,
) -> dict:
    """
    Tính context_precision, faithfulness, answer_relevancy cho Reranker + Generator stage.

    INPUT:
        question             : câu hỏi của test case
        post_rerank_contexts : List[str] — contents của chunks SAU rerank
        answer               : generated_answer từ LLM
        ground_truth         : golden_answer từ qa_pairs.json
        llm                  : LangChain ChatModel instance
        embeddings           : LangChain Embeddings instance

    OUTPUT:
        {
            "context_precision":  float,
            "faithfulness":       float,
            "answer_relevancy":   float
        }
    Nếu lỗi: các field tương ứng = None
    """
    from ragas import evaluate
    from ragas.metrics import context_precision, faithfulness, answer_relevancy
    from datasets import Dataset

    try:
        dataset = Dataset.from_dict({
            'question':     [question],
            'contexts':     [post_rerank_contexts],
            'answer':       [answer],
            'ground_truth': [ground_truth],
        })
        result = evaluate(
            dataset,
            metrics=[context_precision, faithfulness, answer_relevancy],
            llm=_build_ragas_llm(llm),
            embeddings=_build_ragas_embeddings(embeddings),
            raise_exceptions=True,
        )
        def _safe_float(v):
            if isinstance(v, list): v = v[0] if v else None
            try: return float(v) if v is not None else None
            except (TypeError, ValueError): return None
        return {
            'context_precision': _safe_float(result['context_precision']),
            'faithfulness':      _safe_float(result['faithfulness']),
            'answer_relevancy':  _safe_float(result['answer_relevancy']),
        }
    except Exception as e:
        logger.warning(f'[RAGAs] reranker/generator metrics failed: {e}')
        return {
            'context_precision': None,
            'faithfulness':      None,
            'answer_relevancy':  None,
        }
