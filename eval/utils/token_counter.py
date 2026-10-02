"""
eval/utils/token_counter.py
============================
Đếm token bằng tiktoken, không cần gọi API.

INPUT : text: str  hoặc  messages: list[dict]
OUTPUT: số token (int), ước tính dựa trên cl100k_base encoding

Lý do dùng tiktoken thay vì đọc response.usage:
- OpenRouter/free không đảm bảo trả về usage field
- Cho phép pre-count token trước khi gửi request
- Nhanh, local, không tốn chi phí
"""
import tiktoken

# cl100k_base = encoding của GPT-4 / gpt-3.5-turbo / text-embedding-3
# Dùng làm proxy cho mọi LLM qua OpenRouter
_ENCODER = tiktoken.get_encoding('cl100k_base')


def _normalize_to_str(text) -> str:
    """Chuan hoa output tu bat ky LLM nao ve str.
    OpenAI/OpenRouter: content la str.
    Gemini (ChatGoogleGenerativeAI): content co the la list[dict] voi key text.
    """
    if not text:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", "") or part.get("content", ""))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p)
    return str(text)


def count_tokens(text) -> int:
    """Dem so token. Chap nhan str hoac list (Gemini content parts)."""
    normalized = _normalize_to_str(text)
    if not normalized:
        return 0
    return len(_ENCODER.encode(normalized))


def count_messages_tokens(messages: list) -> int:
    """
    Đếm tổng số token của một list messages [{role, content}].
    Áp dụng công thức overhead của OpenAI: +3 tokens/message, +3 cho reply.

    INPUT : messages = [{"role": "system", "content": "..."}, ...]
    OUTPUT: số token ước tính (int)
    """
    total = 3   # reply overhead
    for msg in messages:
        total += 3
        total += count_tokens(msg.get('content', ''))
        total += count_tokens(msg.get('role', ''))
    return total
