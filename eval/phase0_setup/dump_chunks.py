"""
eval/phase0_setup/dump_chunks.py
==================================
PHASE 0b: Đọc FAISS index và xuất toàn bộ chunks ra file JSON.

INPUT  : eval/data/faiss_eval_index/ (phải chạy embed_and_index.py trước)
OUTPUT : eval/data/chunks_dump.json

Chạy lệnh:
    python3 -m eval.phase0_setup.dump_chunks

Mục đích:
    File JSON này cho phép human đọc nội dung từng chunk để:
    1. Thiết kế câu hỏi phù hợp với nội dung thật của tài liệu
    2. Xác định golden_chunk_ids cho mỗi câu hỏi (cho Hit Rate + MRR)
    3. Viết golden_answer chính xác

Format output (chunks_dump.json):
    [
        {
            "chunk_id"   : "chunk_0000",    ← ID dùng trong qa_pairs.json
            "chunk_index": 0,
            "content"    : "# Chương I...", ← Toàn bộ nội dung chunk
            "metadata"   : {"Header 1": "Chương I", ...},
            "preview"    : "150 ký tự đầu để đọc nhanh"
        },
        ...
    ]

Hướng dẫn đọc chunks_dump.json:
    - Mở bằng VS Code + JSON formatter extension để dễ đọc
    - Tìm theo từ khóa trong "preview" để locate chunks liên quan
    - Ghi chunk_id vào golden_chunk_ids trong qa_pairs.json
"""
import sys, os

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

import json
import pickle

from langchain_community.vectorstores import FAISS
from app.model import embeddings

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)
from eval.config import FAISS_INDEX_DIR, CHUNKS_DUMP_PATH


def run():
    index_faiss = os.path.join(FAISS_INDEX_DIR, 'index.faiss')
    index_pkl   = os.path.join(FAISS_INDEX_DIR, 'index.pkl')

    if not os.path.exists(index_faiss):
        raise FileNotFoundError(
            f'\n[ERROR] Không tìm thấy FAISS index tại: {FAISS_INDEX_DIR}\n'
            f'Hãy chạy embed_and_index.py trước.'
        )

    # Đọc Documents từ FAISS pickle
    # FAISS lưu 2 objects: (InMemoryDocstore, index_to_docstore_id)
    # InMemoryDocstore._dict = {uuid: Document}
    with open(index_pkl, 'rb') as f:
        raw = pickle.load(f)

    docs = list(raw[0]._dict.values())
    print(f'[dump] Tìm thấy {len(docs)} chunks trong FAISS index')

    # Sort theo chunk_index để dễ đọc theo thứ tự tài liệu
    docs.sort(key=lambda d: d.metadata.get('chunk_index', 0))

    chunks_data = []
    for doc in docs:
        idx      = doc.metadata.get('chunk_index', 0)
        chunk_id = f'chunk_{idx:04d}'

        chunks_data.append({
            'chunk_id':    chunk_id,
            'chunk_index': idx,
            'content':     doc.page_content,
            'metadata':    {k: v for k, v in doc.metadata.items()
                            if k not in ('chunk_index', 'total_chunk')},
            'preview':     doc.page_content[:150].replace('\n', ' ').strip(),
        })

    with open(CHUNKS_DUMP_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f'[dump] Đã xuất {len(chunks_data)} chunks ra: {CHUNKS_DUMP_PATH}')
    print()
    print('[dump] Tiếp theo:')
    print('  1. Mở eval/data/chunks_dump.json để đọc nội dung tài liệu')
    print('  2. Tạo eval/data/qa_pairs.json với 17 test cases')
    print('  3. Chạy: python3 -m eval.phase1_label.label_assistant')


if __name__ == '__main__':
    run()
