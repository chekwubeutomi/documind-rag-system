# DocMind RAG - Document Retrieval-Augmented Generation System

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) system that combines document retrieval with LLM generation to provide accurate, sourced answers to user queries.

## 🌟 Features

- **Document Upload & Processing**: Support for PDF, Markdown, and Text files
- **Intelligent Chunking**: Configurable text chunking with overlap for better context
- **Vector Embeddings**: Using Sentence Transformers for high-quality semantic embeddings
- **Multiple Vector Databases**: Support for Qdrant and ChromaDB
- **Multiple LLM Providers**: Groq, Ollama, and OpenAI support
- **REST API**: FastAPI-based API with automatic OpenAPI documentation
- **Web UI**: Streamlit-based user interface
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Docker Support**: Complete Docker and docker-compose setup
- **Testing**: Comprehensive unit and integration tests
- **CI/CD**: GitHub Actions workflow for automated testing and deployment

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
└─────────────┬───────────────────────────────┬───────────┘
              │                               │
              ▼                               ▼
        ┌─────────────┐                ┌──────────────┐
        │  Document   │                │   RAG Query  │
        │   Upload    │                │   Interface  │
        └──────┬──────┘                └──────┬───────┘
               │                              │
               └──────────┬───────────────────┘
                          ▼
              ┌─────────────────────────┐
              │   FastAPI Backend       │
              │  - Document Processing  │
              │  - Embedding Service    │
              │  - Query Orchestration  │
              └─────────────────────────┘
                   │              │              │
        ┌──────────▼──┐   ┌──────▼────────┐   ┌─▼──────────────┐
        │  Embedding   │   │ Vector DB     │   │  LLM Provider  │
        │  (Sentence   │   │ (Qdrant/      │   │ (Groq/Ollama/  │
        │  Transformers)   │  ChromaDB)    │   │  OpenAI)       │
        └──────────────┘   └───────────────┘   └────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional, for containerized deployment)
- API keys for LLM providers (Groq, OpenAI, etc.)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/docmind.git
   cd docmind
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   
   **Option 1: API only**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   **Option 2: With Streamlit frontend**
   ```bash
   # Terminal 1 - API
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   
   # Terminal 2 - Frontend
   streamlit run frontend/streamlit_app.py
   ```

### Docker Deployment

1. **Set up environment**
   ```bash
   cp .env.example .env
   # Configure your LLM API keys and other settings
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **Access services**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Frontend: http://localhost:8501
   - Qdrant: http://localhost:6333
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

## 📖 Usage

### API Endpoints

#### Health Check
```bash
GET /api/v1/health
```

#### Upload Document
```bash
POST /api/v1/documents/upload

Content-Type: multipart/form-data
file: <document file>
```

#### Query RAG System
```bash
POST /api/v1/query

{
  "query": "What is FastAPI?",
  "top_k": 5,
  "include_sources": true,
  "stream": false
}
```

### Ingesting Documents

```bash
python scripts/ingest_documents.py ./data/raw --collection documents
```

### Evaluating RAG System

```bash
python scripts/evaluate_rag.py ./data/evaluation/test_questions.json --api-url http://localhost:8000
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest tests/ --cov=app
```

## ⚙️ Configuration

### Environment Variables

Key configuration variables (see `.env.example` for all options):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (groq, ollama, openai) | groq |
| `GROQ_API_KEY` | Groq API key | - |
| `VECTOR_DB_TYPE` | Vector DB type (qdrant, chroma) | qdrant |
| `VECTOR_DB_URL` | Vector DB URL | http://localhost:6333 |
| `CHUNK_SIZE` | Document chunk size | 512 |
| `CHUNK_OVERLAP` | Chunk overlap | 50 |
| `TOP_K_RETRIEVAL` | Number of documents to retrieve | 5 |

## 📁 Project Structure

```
docmind/
├── app/
│   ├── api/                    # API layer
│   │   ├── v1/
│   │   │   ├── endpoints.py   # API endpoints
│   │   │   └── __init__.py
│   │   └── main.py            # API setup
│   ├── core/
│   │   ├── config.py          # Configuration
│   │   ├── logging.py         # Logging setup
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── request_response.py # Pydantic models
│   │   └── __init__.py
│   ├── services/              # Business logic
│   │   ├── document_processor.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── vector_db_service.py
│   │   └── __init__.py
│   └── main.py                # Main FastAPI app
├── frontend/
│   └── streamlit_app.py       # Streamlit UI
├── scripts/
│   ├── ingest_documents.py    # Document ingestion
│   └── evaluate_rag.py        # RAG evaluation
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── conftest.py            # Pytest fixtures
├── data/
│   ├── raw/                   # Original documents
│   ├── processed/             # Processed chunks
│   └── evaluation/            # Test questions
├── models/                    # Model cache
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔄 Workflow

### Document Ingestion Flow
1. User uploads document via API or Streamlit
2. Document processor extracts text (PDF, MD, TXT)
3. Text is chunked with configurable size and overlap
4. Embedding service generates semantic embeddings
5. Vectors stored in vector database with metadata

### Query Flow
1. User submits query via API or Streamlit
2. Query is embedded using the same embedding model
3. Vector DB retrieves top-K similar documents
4. Retrieved documents are ranked by similarity
5. Context is constructed from top documents
6. LLM generates answer based on context
7. Response returned with sources and metadata

## 🛠️ Development

### Running Tests
```bash
# All tests with verbose output
pytest tests/ -v

# Specific test file
pytest tests/unit/test_embedding_service.py -v

# With coverage report
pytest tests/ --cov=app --cov-report=html
```

### Code Quality
```bash
# Format code
black app scripts tests

# Lint code
flake8 app scripts tests

# Type checking
mypy app --ignore-missing-imports
```

## 📊 Monitoring

### Prometheus Metrics
- Request count and latency
- Document processing metrics
- Embedding generation metrics
- Vector DB operations

### Grafana Dashboards
- System health and performance
- API metrics
- Document processing statistics

Access Grafana at `http://localhost:3000` (default: admin/admin)

## 🚀 Deployment

### Production Checklist
- [ ] Set `DEBUG=False` in environment
- [ ] Configure proper API keys for LLM providers
- [ ] Set up vector database with persistence
- [ ] Configure logging to files
- [ ] Set up monitoring and alerting
- [ ] Use Gunicorn/production ASGI server
- [ ] Set up SSL/TLS certificates
- [ ] Configure CORS for frontend domain
- [ ] Set up database backups

### Production Deployment
```bash
# Using Gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker with reverse proxy (nginx)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📚 Available LLM Providers

### Groq (Recommended for fast inference)
```python
LLM_PROVIDER=groq
GROQ_API_KEY=<your-key>
GROQ_MODEL=mixtral-8x7b-32768
```

### Ollama (Local inference)
```python
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### OpenAI
```python
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-3.5-turbo
```

## 🔒 Security Considerations

- Store API keys in environment variables, not in code
- Use `.env` files locally (add to `.gitignore`)
- Enable authentication for production API endpoints
- Validate and sanitize file uploads
- Use HTTPS/TLS in production
- Implement rate limiting
- Set up proper logging and monitoring

## 🐛 Troubleshooting

### Vector DB Connection Issues
```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Verify connection in logs
docker logs docmind-qdrant
```

### LLM Provider Issues
```bash
# Check API key is set
echo $GROQ_API_KEY

# Test LLM connection
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}'
```

### Embedding Service Issues
```bash
# Check embedding model is loaded
curl http://localhost:8000/api/v1/health

# Check logs for download progress
docker logs docmind-api
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact & Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@docmind.ai

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Groq Documentation](https://console.groq.com/docs)

---

**Built with ❤️ using FastAPI, Streamlit, and modern ML tools**