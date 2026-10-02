"""
eval/phase0_setup/embed_and_index.py
======================================
PHASE 0a: Embed tài liệu test và lưu vào FAISS index riêng.

INPUT  : File PDF trong eval/data/document/
OUTPUT : eval/data/faiss_eval_index/index.faiss
         eval/data/faiss_eval_index/index.pkl

Chạy lệnh (từ thư mục RAG_Chatbot_V2/):
    python3 -m eval.phase0_setup.embed_and_index

Lưu ý:
    - Chỉ cần chạy 1 lần duy nhất
    - Nếu thay đổi document → xóa faiss_eval_index/ và chạy lại
    - Docling convert PDF → Markdown: ~3-10 phút tùy kích thước tài liệu
    - Embedding API call: ~1-2 phút
"""
import sys, os

# Thêm project root vào sys.path để import từ app/
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

import glob
import pickle

from langchain_community.vectorstores import FAISS

# Import từ app (cần .env hợp lệ với API_KEY)
from app.model import embeddings
from app.service.RAG_services.ChunkSplitters.HierarchicalChunk import HierarchicalChunk

# Import config eval
EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)
from eval.config import DOCUMENT_DIR, FAISS_INDEX_DIR


def find_document() -> str:
    """Tìm file PDF đầu tiên trong thư mục document."""
    pdfs = glob.glob(os.path.join(DOCUMENT_DIR, '*.pdf'))
    if not pdfs:
        raise FileNotFoundError(
            f'\n[ERROR] Không tìm thấy file PDF trong: {DOCUMENT_DIR}\n'
            f'Hãy đặt file PDF vào thư mục đó rồi chạy lại.'
        )
    if len(pdfs) > 1:
        print(f'[WARN] Tìm thấy {len(pdfs)} PDFs, sử dụng: {os.path.basename(pdfs[0])}')
    return pdfs[0]


def run():
    doc_path = find_document()
    print(f'[embed] Document: {doc_path}')

    # Bước 1: Chia chunk bằng HierarchicalChunk (giống production)
    #   is_complicated_file=True → dùng Docling để convert PDF → Markdown → split theo headers
    #   is_complicated_file=False → dùng RecursiveCharacterTextSplitter (cho file đơn giản)
    chunker = HierarchicalChunk()
    print('[embed] Đang chia chunk (Docling convert PDF → Markdown → split)...')
    print('[embed] Quá trình này mất ~3-10 phút. Vui lòng chờ...')

    chunks = chunker.do_split(
        is_complicated_file=True,
        upload_file_path=doc_path,
        # Eval không có file_id hay db session → truyền None
        # HierarchicalChunk.do_split chỉ dùng upload_file_path khi is_complicated_file=True
        file_id=None,
        db=None,
    )

    print(f'[embed] Tổng số chunks: {len(chunks)}')
    if not chunks:
        raise ValueError(
            '[ERROR] Không có chunk nào được tạo ra.\n'
            'Kiểm tra lại file PDF hoặc Docling installation.'
        )

    # Bước 2: Thêm metadata chunk_index (giống parse_file_and_save_FAISS trong production)
    # chunk_id format: "chunk_XXXX" (4 chữ số, zero-padded)
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
        chunk.metadata['total_chunk'] = len(chunks)

    # Bước 3: Tạo FAISS index từ chunks + embeddings
    print('[embed] Đang tạo FAISS index (gọi API embedding)...')
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Bước 4: Lưu ra disk
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f'[embed] FAISS index đã lưu tại: {FAISS_INDEX_DIR}')
    print('[embed] Hoàn thành! Tiếp theo chạy:')
    print('    python3 -m eval.phase0_setup.dump_chunks')


if __name__ == '__main__':
    run()
