# RAG Chatbot – AI Powered Document Question Answering

A **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask natural language questions about documents. The system retrieves relevant document sections using **semantic search** and generates accurate responses using a large language model.

This project combines **document retrieval, vector search, and conversational AI** to create an intelligent document assistant.

---

# Features

🔗 **Document Integration**
Load and analyze any public document link and convert it into searchable knowledge.

🔍 **Semantic Search**
Uses **TF-IDF embeddings** with **FAISS vector indexing** for fast and accurate retrieval of document chunks.

💬 **AI Generated Responses**
Uses **Llama 3.1-8B** through the Groq API to generate contextual answers.

📚 **Conversation Memory**
Maintains the previous **5 conversation turns** to support follow-up questions.

📝 **Automatic Citations**
Responses include references to document sections such as `[Chunk X]`.

🎨 **Web Interface**
Simple frontend built with **HTML, CSS, and JavaScript**.

⚡ **Production Ready Backend**
Backend built with **FastAPI** including error handling and REST API endpoints.

---

# Quick Start

## Prerequisites

* Python **3.9 or higher**
* Groq API Key

Create a free API key at:

```
https://console.groq.com
```

---

# Installation

Clone the repository and set up the environment.

```bash
git clone <repository-url>
cd RAG-Chatbot
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env
```

Add your Groq API key inside `.env`

```
GROQ_API_KEY=your_api_key_here
```

---

# Running the Application

Development mode:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Alternative method:

```bash
python app/main.py
```

Open the application in your browser:

```
http://127.0.0.1:8000
```

---

# System Architecture

The system is composed of several modules responsible for document processing, retrieval, and response generation.

## Core Components

**FastAPI Application**

Handles REST API requests and serves the web interface.

```
app/main.py
```

**Document Loader**

Fetches document content using export APIs.

```
app/gdoc_loader.py
```

**Text Chunker**

Splits the document into overlapping chunks for better retrieval.

```
app/chunker.py
```

**RAG Engine**

Responsible for embeddings, retrieval, and LLM generation.

```
app/rag.py
```

**Memory Manager**

Maintains session state and conversation history.

```
app/memory.py
```

---

# Data Flow

Document Processing Pipeline

```
Document Link
      ↓
Fetch Document
      ↓
Text Chunking
      ↓
TF-IDF Embeddings
      ↓
FAISS Vector Index
```

Query Processing

```
User Question
      ↓
Retrieve Top Relevant Chunks
      ↓
Combine Context + Conversation Memory
      ↓
Generate Answer with LLM
      ↓
Return Response
```

Conversation Handling

```
User Query
      ↓
Store Query + Response
      ↓
Maintain Sliding Window (5 Turns)
```

---

# API Endpoints

| Endpoint                    | Method | Description                     |
| --------------------------- | ------ | ------------------------------- |
| `/health`                   | GET    | System health check             |
| `/api/load-document`        | POST   | Load and process document       |
| `/api/query`                | POST   | Query the loaded document       |
| `/api/clear-session`        | POST   | Reset document and conversation |
| `/api/conversation-history` | GET    | Retrieve chat history           |

---

# Technical Details

## Text Processing

Chunking strategy:

* Word based segmentation
* Approximately **15 chunks per document**
* **10% overlap** between chunks

Embeddings:

* TF-IDF vectorization
* L2 normalization

Search Engine:

* FAISS `IndexFlatL2`
* Cosine similarity retrieval

Top results:

```
Top 5 most relevant chunks
```

---

# Language Model Configuration

Model used:

```
Llama 3.1-8B
```

Parameters:

```
Temperature: 0.7
Max Tokens: 500
```

Context includes:

* Retrieved document chunks
* Conversation history

---

# Conversation Memory

Session Storage:

* Stored in memory during runtime

Conversation Handling:

* Sliding window of **5 turns**

Persistence:

* Data resets when the server restarts

---

# Project Dependencies

| Package       | Purpose                         |
| ------------- | ------------------------------- |
| fastapi       | Web framework                   |
| uvicorn       | ASGI server                     |
| scikit-learn  | TF-IDF embeddings               |
| faiss-cpu     | Vector similarity search        |
| requests      | HTTP requests                   |
| python-dotenv | Environment variable management |

---

# Production Deployment

Install production server:

```bash
pip install gunicorn
```

Run the application:

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Deployment considerations:

* Single user session architecture
* In-memory storage
* Use reverse proxy for HTTPS
* Monitor system via `/health` endpoint

---

# Example Usage

Example using Python requests.

Load document:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/load-document",
    json={"google_docs_link": "https://docs.google.com/document/d/..."}
)
```

Query document:

```python
response = requests.post(
    "http://127.0.0.1:8000/api/query",
    json={"query": "What is the main topic of the document?"}
)
```

---

# Important Notes

* Only **public documents** are supported
* All processing occurs **in memory**
* Restarting the server clears session data
* API rate limits may apply
* Responses include citations such as:

```
[Chunk X]
```

