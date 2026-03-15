"""
FastAPI backend for RAG Chatbot.
Main application with endpoints for document loading, querying, and health checks.
"""

import logging
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.gdoc_loader import load_google_doc
from app.chunker import chunk_text
from app.rag import RAGEngine, extract_chunk_ids_from_response
from app.memory import SessionMemory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot",
    description="Production-ready private RAG chatbot with Google Docs integration",
    version="1.0.0"
)

# Initialize session state
session_memory = SessionMemory()
rag_engine = RAGEngine()

# Pydantic models for request/response
class LoadDocumentRequest(BaseModel):
    """Request to load a Google Doc."""
    google_docs_link: str

class LoadDocumentResponse(BaseModel):
    """Response after loading a Google Doc."""
    success: bool
    message: str
    doc_id: str = None
    chunk_count: int = None

class QueryRequest(BaseModel):
    """Request to query the document."""
    query: str

class QueryResponse(BaseModel):
    """Response to a query."""
    answer: str
    chunks_used: list
    conversation_turn: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    document_loaded: bool
    chunk_count: int = None


# Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns: Application status and loaded document info
    """
    try:
        doc_count = len(session_memory.chunks) if session_memory.is_document_loaded() else None
        response = {
            "status": "healthy",
            "document_loaded": session_memory.is_document_loaded(),
            "chunk_count": doc_count
        }
        logger.debug(f"Health check response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/api/load-document", response_model=LoadDocumentResponse)
async def load_document(request: LoadDocumentRequest):
    """
    Load a Google Doc for RAG.
    
    Args:
        request: Contains google_docs_link
        
    Returns:
        LoadDocumentResponse with doc_id and chunk_count
        
    Raises:
        HTTPException: If document loading fails
    """
    try:
        logger.info(f"Loading Google Doc from link")
        
        # Load document from Google Docs
        doc_text, doc_id = load_google_doc(request.google_docs_link)
        logger.info(f"Retrieved document with {len(doc_text)} characters")
        
        # Chunk the document
        logger.info(f"Chunking document...")
        chunks = chunk_text(doc_text, target_chunk_count=15)
        
        if not chunks:
            logger.error("No chunks created from document")
            raise ValueError("Could not create chunks from document")
        
        logger.info(f"Successfully created {len(chunks)} chunks")
        
        # Create embeddings
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        rag_engine.create_embeddings(chunks)
        logger.info(f"Embeddings created successfully")
        
        # Store in session
        session_memory.load_document(doc_id, doc_text, chunks)
        session_memory.set_embeddings(rag_engine.vectorizer, rag_engine.faiss_index)
        
        # Reset conversation for new document
        session_memory.conversation.clear()
        
        logger.info(f"Successfully loaded document with {len(chunks)} chunks")
        
        return {
            "success": True,
            "message": f"Document loaded successfully with {len(chunks)} chunks",
            "doc_id": doc_id,
            "chunk_count": len(chunks)
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load document: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the loaded document using RAG.
    
    Args:
        request: Contains user query
        
    Returns:
        QueryResponse with answer and chunks used
        
    Raises:
        HTTPException: If no document is loaded or query fails
    """
    try:
        if not session_memory.is_document_loaded():
            raise ValueError("No document loaded. Please load a Google Doc first.")
        
        user_query = request.query.strip()
        
        if not user_query:
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Retrieve relevant chunks
        retrieved_chunks = rag_engine.retrieve_chunks(user_query, k=5)
        
        if not retrieved_chunks:
            logger.warning("No relevant chunks found for query")
            answer = "I couldn't find relevant information in the document to answer this question."
            chunks_used = []
        else:
            # Format context
            context = rag_engine.format_context(retrieved_chunks)
            
            # Get conversation history
            conv_history = session_memory.conversation.get_context_string()
            
            # Generate response
            answer = rag_engine.generate_response(
                query=user_query,
                context=context,
                conversation_history=conv_history,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract chunk IDs from response
            chunks_used = extract_chunk_ids_from_response(answer)
        
        # Store in conversation memory
        session_memory.conversation.add_turn(user_query, answer, chunks_used)
        
        logger.info(f"Query processed successfully. Chunks used: {chunks_used}")
        
        return {
            "answer": answer,
            "chunks_used": chunks_used,
            "conversation_turn": len(session_memory.conversation.get_turns())
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/api/clear-session")
async def clear_session():
    """
    Clear the current session (document and conversation history).
    
    Returns:
        Success message
    """
    try:
        session_memory.clear()
        logger.info("Session cleared")
        return {"success": True, "message": "Session cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@app.get("/api/conversation-history")
async def get_conversation_history():
    """
    Get the current conversation history.
    
    Returns:
        List of conversation turns
    """
    try:
        turns = session_memory.conversation.get_turns()
        return {
            "turns": turns,
            "turn_count": len(turns),
            "max_turns": session_memory.conversation.max_turns
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")


# Serve static files and templates
@app.get("/")
async def root():
    """Serve the main chat interface."""
    return FileResponse("templates/index.html")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get config from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
