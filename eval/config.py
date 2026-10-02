"""
eval/config.py
==============
Central config cho toàn bộ eval pipeline.
Tất cả paths và constants được định nghĩa ở đây.
"""
import os
from dotenv import load_dotenv

# Load .env tu project root truoc khi doc os.environ
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(_env_path, override=True)


# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT     = os.path.abspath(os.path.join(EVAL_DIR, '..'))

DATA_DIR         = os.path.join(EVAL_DIR, 'data')
DOCUMENT_DIR     = os.path.join(DATA_DIR, 'document')
FAISS_INDEX_DIR  = os.path.join(DATA_DIR, 'faiss_eval_index')
CHUNKS_DUMP_PATH = os.path.join(DATA_DIR, 'chunks_dump.json')
QA_PAIRS_PATH    = os.path.join(DATA_DIR, 'qa_pairs.json')
RESULTS_DIR      = os.path.join(EVAL_DIR, 'results')

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K  = 10   # Số chunks trả về từ FAISS + BM25 (trước rerank)
RERANK_TOP_K     = 5    # Số chunks giữ lại sau rerank

# ── RAGAs ────────────────────────────────────────────────────────────────────
# RAGAs sẽ dùng app.model.llm và app.model.embeddings mặc định.
# Nếu muốn dùng Gemini làm judge:
#   RAGAS_JUDGE_MODEL = "gemini/gemini-1.5-pro" (set trong .env hoặc ở đây)
# Xem ragas_wrapper.py để biết cách cấu hình.
RAGAS_RAISE_EXCEPTIONS = False   # True để debug, False để production

# ── Hallucination Judge ────────────────────────────────────────────────────────
# NOTE: Để giảm self-judge bias, bạn nên dùng model KHÁC với generator.
# Cách dùng Gemini làm judge:
#   1. Thêm GEMINI_API_KEY vào .env
#   2. Set JUDGE_MODEL = "gemini-1.5-flash" (hoặc "gemini-1.5-pro")
#   3. collect.py sẽ tự khởi tạo judge_llm riêng nếu GEMINI_API_KEY tồn tại
# Nếu không có GEMINI_API_KEY → fallback dùng cùng llm (mặc định production behavior)
JUDGE_MODEL      = os.environ.get('JUDGE_MODEL', None)       # None = dùng llm mặc định
GEMINI_API_KEY   = os.environ.get('GEMINI_API_KEY', None)

# ── Semantic Similarity ───────────────────────────────────────────────────────
SEM_SIM_MODEL    = 'all-MiniLM-L6-v2'   # Local, không cần API
SEM_SIM_THRESHOLD = 0.70                 # >= threshold = pass

# ── Directories ───────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCUMENT_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
