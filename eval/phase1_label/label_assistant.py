"""
eval/phase1_label/label_assistant.py
=======================================
PHASE 1: LLM gợi ý golden_chunk_ids cho từng test case.

INPUT  : eval/data/chunks_dump.json   (từ phase0b)
         eval/data/qa_pairs.json      (template với question + golden_answer đã điền)
OUTPUT : eval/data/qa_pairs.json      (cập nhật thêm field label_suggestions)

Chạy lệnh:
    python3 -m eval.phase1_label.label_assistant

Quy trình:
    1. Đọc toàn bộ chunks từ chunks_dump.json
    2. Với mỗi question trong qa_pairs chưa có golden_chunk_ids:
       a. Gửi LLM danh sách chunk previews + câu hỏi
       b. LLM gợi ý: chunk nào chứa thông tin trả lời câu hỏi?
       c. Lưu suggestions vào field "label_suggestions" trong qa_pairs.json
    3. HUMAN tự đọc suggestions, verify, và điền vào "golden_chunk_ids"

SAU KHI CHẠY:
    Mở eval/data/qa_pairs.json, với mỗi câu hỏi:
    1. Đọc label_suggestions.relevant_chunk_ids
    2. Tra cứu chunk đó trong chunks_dump.json để xem full content
    3. Nếu đúng: copy vào golden_chunk_ids
    4. Nếu sai/thiếu: tự tìm thêm từ chunks_dump.json
    5. Xóa field label_suggestions sau khi đã confirm (optional)
"""
import sys, os, json, re

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, '../..'))
sys.path.insert(0, PROJECT_ROOT)

EVAL_PACKAGE = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_PACKAGE)

from app.model import llm
from app.service.LLM_service import llm_service
from eval.config import CHUNKS_DUMP_PATH, QA_PAIRS_PATH


LABEL_SYSTEM_PROMPT = (
    'Bạn là chuyên gia phân tích tài liệu pháp lý. '
    'Nhiệm vụ: xác định chunk nào trong tài liệu chứa thông tin cần thiết để trả lời câu hỏi. '
    'Chỉ chọn chunk TRỰC TIẾP chứa thông tin, không chọn chunk chỉ liên quan gián tiếp.'
)

LABEL_USER_TEMPLATE = """DANH SÁCH CHUNKS (preview 150 ký tự đầu mỗi chunk):
{chunks_text}

CÂU HỎI: {question}

Trả lời đúng định dạng JSON (không thêm markdown code block):
{{
  "relevant_chunk_ids": ["chunk_XXXX", "chunk_YYYY"],
  "reasoning": "Giải thích ngắn tại sao chọn các chunks này"
}}"""


def format_chunks_for_prompt(chunks: list, max_chunks: int = 40) -> str:
    """Format chunk list thành text để đưa vào prompt."""
    lines = []
    for c in chunks[:max_chunks]:
        lines.append(f"- {c['chunk_id']}: {c['preview']}")
    if len(chunks) > max_chunks:
        lines.append(f'... và {len(chunks) - max_chunks} chunks nữa (ẩn để giảm token)')
    return '\n'.join(lines)


def suggest_chunks(chunks: list, question: str) -> dict:
    """Gửi LLM gợi ý golden chunk IDs cho 1 câu hỏi."""
    chunks_text = format_chunks_for_prompt(chunks)
    user_content = LABEL_USER_TEMPLATE.format(
        chunks_text=chunks_text,
        question=question,
    )
    # Dùng hallucination_check task (system prompt skeptical — phù hợp cho labeling task)
    response = llm_service.ask_model(llm, 'hallucination_check', user_content)
    raw = response.content.strip()

    # Strip markdown code block nếu LLM thêm vào
    if raw.startswith('```'):
        raw = '\n'.join(raw.split('\n')[1:-1])

    # Parse JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        'relevant_chunk_ids': [],
        'reasoning': f'Parse lỗi. Raw response: {raw[:300]}'
    }


def run():
    # Load chunks
    if not os.path.exists(CHUNKS_DUMP_PATH):
        raise FileNotFoundError(
            f'Không tìm thấy: {CHUNKS_DUMP_PATH}\n'
            f'Chạy phase0b trước: python3 -m eval.phase0_setup.dump_chunks'
        )
    with open(CHUNKS_DUMP_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f'[label] Loaded {len(chunks)} chunks')

    # Load qa_pairs
    if not os.path.exists(QA_PAIRS_PATH):
        raise FileNotFoundError(
            f'Không tìm thấy: {QA_PAIRS_PATH}\n'
            f'Tạo file qa_pairs.json với các test cases trước.'
        )
    with open(QA_PAIRS_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    print(f'[label] Loaded {len(qa_pairs)} questions')

    # Label từng câu hỏi chưa có golden_chunk_ids
    updated = False
    for i, qa in enumerate(qa_pairs, 1):
        existing = qa.get('golden_chunk_ids', [])
        # Skip nếu đã có chunk IDs thực (không phải placeholder)
        has_real_ids = existing and not any('FILL' in str(c) or 'TODO' in str(c) for c in existing)
        if has_real_ids:
            print(f'[label] [{i}/{len(qa_pairs)}] {qa["id"]}: skip (đã có golden_chunk_ids)')
            continue

        print(f'[label] [{i}/{len(qa_pairs)}] {qa["id"]}: labeling...')
        suggestion = suggest_chunks(chunks, qa['question'])
        qa['label_suggestions'] = suggestion
        updated = True
        print(f'  → Gợi ý: {suggestion.get("relevant_chunk_ids", [])}')
        print(f'  → Lý do: {suggestion.get("reasoning", "")[:100]}')

    if updated:
        with open(QA_PAIRS_PATH, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f'\n[label] Đã lưu suggestions vào: {QA_PAIRS_PATH}')
        print()
        print('[label] VIỆC CỦA HUMAN SAU KHI CHẠY:')
        print('  1. Mở eval/data/qa_pairs.json')
        print('  2. Với mỗi câu hỏi, đọc label_suggestions.relevant_chunk_ids')
        print('  3. Tra cứu chunk đó trong chunks_dump.json để xem full content')
        print('  4. Nếu đúng: copy sang golden_chunk_ids')
        print('  5. Nếu sai/thiếu: tự tìm thêm từ chunks_dump.json')
    else:
        print('[label] Không có câu hỏi nào cần label.')


if __name__ == '__main__':
    run()
