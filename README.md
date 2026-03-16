# RAG Chatbot v2

An improved RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, featuring advanced chunking strategies, hybrid search, and adaptive query handling.

---

## Overview

RAG Chatbot is a document-based question answering system that enhances LLM responses by retrieving relevant information from uploaded files before generating answers.

**Previous version** supported: file upload, vector database persistence, conversation history, reranking, and logging.

**This version** brings significant improvements in retrieval quality, chunking flexibility, and answer reliability — detailed in the section below.

---

## What's New in v2

| Area | Improvement |
|---|---|
| **File support** | PDF, DOCX, TXT with asynchronous background ingestion |
| **Chunking** | Hierarchical, Semantic, Hybrid strategies — configurable via `.env` |
| **Search** | Hybrid search (Dense + BM25) merged with RRF ranking |
| **Retrieval** | Multi-query retrieval + query decomposition |
| **Adaptive RAG** | Query classification (chit-chat vs document), relevance check, sufficiency check with retry |
| **Reliability** | Hallucination detection with confidence score |
| **Citation** | Inline source attribution `[1]`, `[2]` with file name and page number |
| **Bugs** | Remaining issues from v1 resolved |

---

## Project Structure

```
RAG_mini_project/
│
├── 📄 .gitignore
├── 📄 .env.example
├── 📄 README.md
├── 📄 requirements.txt
│
└── 📁 app/
    ├── 📄 main.py                       # FastAPI entry point
    ├── 📄 model.py                      # LLM & embeddings initialization
    ├── 📄 db.py                         # Database connection
    ├── 📄 deps.py                       # Dependencies & validators
    │
    ├── 📁 api/                          # API layer
    │   ├── 📄 upload.py                 # POST /upload
    │   ├── 📄 chat.py                   # POST /chat
    │   └── 📄 session.py                # Session management
    │
    ├── 📁 core/                         # App configuration
    │   ├── 📄 env_config.py             # Settings from .env
    │   ├── 📄 celery_app.py             # Background task worker
    │   ├── 📄 redis.py                  # Redis configuration
    │   ├── 📄 limiter.py                # Rate limiting
    │   └── 📄 logging_config.py         # Logging setup
    │
    ├── 📁 schemas/                      # Pydantic models
    │   ├── 📄 request_model.py          # Input schemas
    │   └── 📄 response_model.py         # Output schemas
    │
    ├── 📁 prompt/                       # LLM prompt templates
    │   ├── 📄 analyze_query.txt
    │   ├── 📄 direct_answer.txt
    │   ├── 📄 hallucination_check.txt
    │   ├── 📄 question_answer.txt
    │   ├── 📄 summarization.txt
    │   └── 📄 synthesize_context.txt
    │
    └── 📁 service/                      # Business logic
        ├── 📄 DB_service.py
        ├── 📄 LLM_service.py
        ├── 📄 CM_service.py             # Conversation management
        │
        ├── 📁 File_service/             # File processing layer
        │   ├── 📄 base.py               # Abstract base class
        │   ├── 📄 FileFactory.py        # Factory pattern registry
        │   ├── 📄 PDF.py                # PyMuPDF processor
        │   ├── 📄 DOCX.py               # python-docx processor
        │   ├── 📄 TXT.py                # Plain text processor
        │   └── 📄 Docling_service.py    # Complex file converter
        │
        └── 📁 RAG_services/
            ├── 📄 RAG_service.py        # FAISS + BM25 + RRF orchestration
            └── 📁 ChunkSplitters/
                ├── 📄 ChunkBase.py
                ├── 📄 ChunkFactory.py
                ├── 📄 HierarchicalChunk.py
                ├── 📄 SemanticChunk.py
                └── 📄 HybridChunk.py
```

---

## Data Flow

### Upload Flow

```
User File
    │
    ▼
POST /upload
    │
    ▼
validate_file_size() + check_file_type()
    │
    ▼
FileFactory → PDF / DOCX / TXT processor
    │
    ▼
processor.save_file() → uploads/{file_id}_{filename}
    │
    ▼
db_service.create_file() → FileStatus table
    │
    ▼
Celery background task: process_pdf_file_background()
    │
    ├── inspect_file_and_routing() → is_complicated?
    │       ├── YES → Docling_service.convert_docling_to_list_document()
    │       └── NO  → processor.process_file() → List[Document]
    │
    ▼
RAGService.parse_file_and_save_FAISS()
    │
    ├── ChunkFactory.get_registry(strategy)
    │       ├── "hierarchical" → HierarchicalChunk
    │       ├── "semantic"     → SemanticChunk
    │       └── "hybrid"       → HybridChunk
    │
    ├── chunker.do_split() → List[Chunk]
    │
    ├── Update metadata: {file_id, file_name, page, language, chunk_index, ...}
    │
    └── FAISS.from_documents() + save_local()
            │
            ▼
    FileStatus.status = SUCCESS
```

### Chat Flow

```
User Query
    │
    ▼
POST /chat  {file_id, question, session_id}
    │
    ▼
validate_file_available() + validate_file_status()
    │
    ▼
LLMService.analyze_query()
    └── returns: {type, round1: [queries], round2: [queries]}
    │
    ├── type == "chit_chat"
    │       └── LLM answers directly (no retrieval)
    │
    └── type == "simple" | "complex"
            │
            ▼
        RAGService.load_FAISS_and_retrieve()
            ├── Load FAISS + build BM25 from index
            ├── multi_query_hybrid_search()
            │       ├── BM25 search
            │       ├── FAISS semantic search
            │       └── RRF fusion
            │
            ├── relevance_and_sufficiency_check()
            │       └── retry with round2 queries if needed
            │
            └── rerank() via cross-encoder
            │
            ▼
        CM_service.analyze_conversation_history()
            ├── get_last_summary()
            └── compress old dialogs if token > threshold
            │
            ▼
        LLMService.ask_model() with context + citation prompt
            │
            ▼
        LLMService.hallucination_check()
            └── returns confidence score (0.0 - 1.0)
            │
            ▼
        ChatBotResponse
            {model_name, session_id, answer, sources, confidence}
```

---

## Key Components

| Component | Role |
|---|---|
| **FileFactory** | Factory pattern — register and route file processors |
| **PDF / DOCX / TXT** | Extract text, detect complexity, save file |
| **Docling_service** | Convert complex scanned/structured files via Docling |
| **ChunkFactory** | Factory pattern — register and route chunking strategies |
| **HierarchicalChunk** | Split by Markdown headers (requires heading structure) |
| **SemanticChunk** | LLM-based semantic boundary detection |
| **HybridChunk** | Parent-child chunks with parent stored in DB |
| **RAGService** | Core RAG orchestration: FAISS, BM25, RRF, rerank |
| **LLMService** | LLM calls, prompt formatting, hallucination check |
| **CM_service** | Conversation history + rolling summarization |
| **DB_service** | All database CRUD operations |

---

## Database Schema

| Table | Purpose | Key Fields |
|---|---|---|
| **ConversationHistory** | Chat dialogs | `session_id`, `session_name`, `role`, `content`, `id` |
| **Summary** | Compressed conversation summaries | `session_id`, `covered_until_message_id`, `content` |
| **FileStatus** | Uploaded file metadata & processing state | `file_id`, `file_name`, `type`, `status`, `timestamp` |
| **ParentStore** | Parent chunk context for hybrid mode | `parent_id`, `file_id`, `context` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI |
| **LLM** | LangChain + OpenRouter API |
| **Embeddings** | Hugging Face (via `model.py`) |
| **Vector Store** | FAISS |
| **Full-text Search** | BM25 |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **ORM** | SQLModel |
| **Document Processing** | PyMuPDF · python-docx · Docling |
| **Background Tasks** | Celery + Redis |
| **Rate Limiting** | SlowAPI |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a file for processing |
| `POST` | `/chat` | Ask a question against an uploaded file |
| `GET` | `/session/list` | List all conversation sessions |
| `POST` | `/session/update/{session_id}` | Rename a session |
| `DELETE` | `/session/delete/{session_id}` | Delete a session and its history |
