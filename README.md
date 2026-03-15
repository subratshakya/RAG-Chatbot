# RAG Chatbot - Private Document Q&A with AI

A production-ready Retrieval-Augmented Generation (RAG) chatbot that reads from Google Docs and answers questions using Llama 3 via Groq API. Built with FastAPI, FAISS vector search, and TF-IDF embeddings.

## ✨ Features

- **🔗 Google Docs Integration**: Load any public Google Doc with a single link
- **🔍 Smart Retrieval**: TF-IDF embeddings with FAISS for fast semantic search
- **💬 AI-Powered Answers**: Llama 3.1-8B via Groq API for contextual responses
- **📚 Multi-turn Memory**: Maintains last 5 conversation turns for follow-ups
- **📝 Inline Citations**: All responses include chunk references like [Chunk 3]
- **🎨 Web Interface**: Clean, responsive HTML/CSS frontend
- **⚡ Production Ready**: FastAPI backend with comprehensive error handling
- **🏥 Health Checks**: Built-in `/health` endpoint for monitoring

## � Demo Video

Watch the chatbot in action: [Demo Video](https://drive.google.com/file/d/1_BBG8w-AkcTqnq5ya5NIEl72CYArZpGf/view?usp=sharing)

## �📋 Requirements

- Python 3.9+
- pip or conda
- Valid Groq API key (free at https://console.groq.com)

## 🚀 Quick Start

### 1. Clone or Download the Project

```bash
cd RAG_privatebot
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get your free Groq API key at: https://console.groq.com/keys

### 5. Run the Server

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or directly:

```bash
python app/main.py
```

### 6. Access the Web Interface

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## 💡 How to Use

### Loading a Document

1. Find a public Google Doc you want to analyze
2. Share it publicly (Share → Anyone with the link)
3. Copy the link to the "Load a Google Doc" field
4. Click "Load Document"
5. The system will extract, chunk, and index the document

### Asking Questions

1. After loading a document, type your question in the chat
2. The system retrieves relevant chunks using semantic search
3. The question + chunks are sent to Llama 3 for generation
4. Responses include citations to chunks used (e.g., [Chunk 3])
5. Follow-up questions use conversation history for context

### Example Questions

- "What is the main topic of this document?"
- "Summarize the key points"
- "What does it say about [specific topic]?"
- "Compare this with [concept]?"
- "How do I do [task] based on this document?"

## 🏗️ Architecture

```
RAG_privatebot/
├── app/
│   ├── main.py           # FastAPI application & routes
│   ├── gdoc_loader.py    # Google Docs fetching & parsing
│   ├── chunker.py        # Text chunking with overlap
│   ├── rag.py            # FAISS & Groq LLM integration
│   └── memory.py         # Conversation & session memory
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment config
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 API Endpoints

### Health Check
```
GET /health
```
Returns: `{status, document_loaded, chunk_count}`

### Load Document
```
POST /api/load-document
Body: {"google_docs_link": "https://docs.google.com/document/d/..."}
Returns: {success, message, doc_id, chunk_count}
```

### Query Document
```
POST /api/query
Body: {"query": "Your question here"}
Returns: {answer, chunks_used, conversation_turn}
```

### Get Conversation History
```
GET /api/conversation-history
Returns: {turns, turn_count, max_turns}
```

### Clear Session
```
POST /api/clear-session
Returns: {success, message}
```

## 🧠 Technical Details

### Text Chunking
- Word-based chunks with configurable overlap
- Target: 10-25 chunks per document
- Automatic sizing based on document length
- Each chunk maintains a unique ID for citations

### Embeddings & Retrieval
- **Algorithm**: TF-IDF (sklearn)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Similarity Metric**: Cosine similarity (L2 distance)
- **Retrieval**: Top-5 most relevant chunks
- **Processing**: Single request handles end-to-end retrieval

### LLM Integration
- **Model**: Llama 3.1-8B-Instant (via Groq API)
- **Context**: Retrieved chunks + conversation history
- **Temperature**: 0.7 (balanced creativity/consistency)
- **Max Tokens**: 500 per response
- **Citations**: Automatic [Chunk X] formatting

### Memory Management
- **Conversation Window**: Last 5 turns
- **Session State**: Document text, chunks, FAISS index
- **Persistence**: In-memory (cleared on server restart)
- **History Tracking**: Timestamps, chunks used per turn

## 🛡️ Error Handling

The system gracefully handles:

| Error | Handling |
|-------|----------|
| Invalid Google Doc link | Clear error message with expected format |
| Private/inaccessible docs | Helpful message about sharing settings |
| Empty documents | Error indicating document is empty |
| Network timeouts | Retry-friendly timeout messages |
| API failures | Fallback messages without crashing |
| Query not in document | Response indicating insufficient info |
| Rate limits | User-friendly rate limit messages |

## ⚙️ Configuration

### Environment Variables

```env
# Required
GROQ_API_KEY=sk_...your_key_here

# Optional
HOST=127.0.0.1           # Server host (default: 127.0.0.1)
PORT=8000                # Server port (default: 8000)
DEBUG=False              # Debug mode (default: False)
```

### Tuning Parameters

Edit these in `app/chunker.py` for different chunking behavior:

```python
chunk_text(
    text,
    target_chunk_count=15,  # Aim for ~15 chunks
    overlap_ratio=0.1,      # 10% overlap between chunks
    min_chunk_size=50,      # Minimum 50 words
    max_chunk_size=1000     # Maximum 1000 words
)
```

Edit these in `app/rag.py` for retrieval tuning:

```python
retrieve_chunks(
    query,
    k=5,                    # Retrieve top 5 chunks
    score_threshold=0.0     # No minimum similarity
)
```

## 🐛 Troubleshooting

### "GROQ_API_KEY not configured"
- Copy `.env.example` to `.env`
- Add your actual API key from https://console.groq.com
- Restart the server

### "Document is not publicly accessible"
- Share the Google Doc with "Anyone with the link"
- Ensure sharing is set to "Viewer" or higher
- Use the full shareable link

### "Failed to connect to LLM service"
- Check your internet connection
- Verify Groq API is accessible
- Check that your API key is valid

### Empty or minimal responses
- Ensure document is loaded correctly (check chunk count)
- Try rephrasing your question
- Use more specific keywords from the document

### Slow response times
- Larger documents naturally take longer
- FAISS indexing happens at load time
- Groq API response time varies by server load

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| pydantic | 2.5.0 | Data validation |
| requests | 2.31.0 | HTTP client |
| scikit-learn | 1.3.2 | TF-IDF embeddings |
| numpy | 1.26.2 | Numerical computing |
| faiss-cpu | 1.7.4 | Vector similarity search |
| python-dotenv | 1.0.0 | Environment variables |

## 🚀 Production Deployment

For production use:

1. **Set environment variables:**
   ```env
   HOST=0.0.0.0
   PORT=80
   DEBUG=False
   ```

2. **Use a production ASGI server (Gunicorn + Uvicorn):**
   ```bash
   pip install gunicorn
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. **Add SSL/HTTPS** with a reverse proxy (Nginx, Caddy)

4. **Monitor logs** and use the `/health` endpoint for uptime checks

5. **Rate limiting** - Consider adding rate limiting middleware

6. **Persistent storage** - Enhance memory module to use a database

## 📝 Example Usage Script

```python
"""
Example usage of the RAG API
"""
import requests

BASE_URL = "http://127.0.0.1:8000/api"

# Load a document
doc_response = requests.post(
    f"{BASE_URL}/load-document",
    json={"google_docs_link": "https://docs.google.com/document/d/1xxx..."}
)
print(doc_response.json())

# Query the document
query_response = requests.post(
    f"{BASE_URL}/query",
    json={"query": "What is the main topic?"}
)
print(query_response.json())

# Get history
history_response = requests.get(f"{BASE_URL}/conversation-history")
print(history_response.json())
```

## 🔒 Privacy & Security Notes

- **No cloud storage**: All data stays on your local machine
- **Google Docs**: Only public docs can be fetched (you control sharing)
- **API Keys**: Store in `.env`, never commit to git
- **No data logging**: Responses aren't logged or tracked
- **Document privacy**: Use private/local deployment for sensitive docs

## 📈 Performance Metrics

On typical hardware:

- **Document loading**: 1-5 seconds (depends on document size)
- **Embedding creation**: 2-10 seconds for 100KB doc
- **Query retrieval**: 100-500ms (sub-second for most queries)
- **LLM generation**: 2-5 seconds (Groq API latency)
- **Total response time**: 5-15 seconds per query

## 🤝 Contributing

Improvements and bug fixes welcome! Consider:

- Better chunking strategies
- Alternative vector databases (Pinecone, Weaviate)
- Additional LLM providers
- Persistent storage integration
- Docker containerization

## 📄 License

This project is provided as-is for educational and personal use.

## 🆘 Support

- Check the troubleshooting section
- Review error messages in the UI or logs
- Verify your Groq API key is active
- Ensure documents are publicly shared

## 🎯 Roadmap

Future enhancements:

- [ ] Multiple document support
- [ ] Document summarization
- [ ] Export conversation to PDF
- [ ] User authentication
- [ ] Persistent database storage
- [ ] Docker support
- [ ] Alternative embedding models
- [ ] Web search integration
- [ ] Custom prompt templates
- [ ] Performance analytics

---

**Built with ❤️ for efficient AI-powered document analysis**

Last updated: 2026
#   R A G - C h a t b o t  
 