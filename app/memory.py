"""
Conversation memory module for RAG chatbot.
Maintains multi-turn conversation history with configurable retention.
"""

import logging
from typing import List, Dict, Tuple
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history with a fixed window of recent turns.
    """
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep
        """
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)
        self.session_start = datetime.now()
        
    def add_turn(self, user_message: str, assistant_message: str, context_chunks: List[int] = None):
        """
        Add a conversation turn (user message + assistant response).
        
        Args:
            user_message: The user's question/input
            assistant_message: The assistant's response
            context_chunks: List of chunk IDs used for this response
        """
        turn = {
            "user": user_message,
            "assistant": assistant_message,
            "chunks": context_chunks or [],
            "timestamp": datetime.now().isoformat()
        }
        self.turns.append(turn)
        logger.debug(f"Added turn. Memory now has {len(self.turns)} turns")
    
    def get_context_string(self) -> str:
        """
        Get formatted conversation history for context in prompts.
        
        Returns:
            Formatted string of previous turns for inclusion in LLM prompt
        """
        if not self.turns:
            return ""
        
        context_lines = ["## Previous Conversation Context:"]
        for i, turn in enumerate(self.turns, 1):
            context_lines.append(f"\nUser: {turn['user']}")
            context_lines.append(f"Assistant: {turn['assistant']}")
        
        context_lines.append("\n## Current Query:")
        return "\n".join(context_lines)
    
    def get_turns(self) -> List[Dict]:
        """
        Get all stored turns.
        
        Returns:
            List of conversation turn dictionaries
        """
        return list(self.turns)
    
    def clear(self):
        """Clear all conversation history."""
        self.turns.clear()
        logger.info("Conversation memory cleared")
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics about the conversation.
        
        Returns:
            Dictionary with session information
        """
        return {
            "num_turns": len(self.turns),
            "max_turns": self.max_turns,
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "turns": list(self.turns)
        }


class SessionMemory:
    """
    Manages current session state including loaded documents and indices.
    """
    
    def __init__(self):
        """Initialize session memory."""
        self.document_id = None
        self.document_text = None
        self.chunks = []
        self.embeddings_index = None
        self.tfidf_vectorizer = None
        self.conversation = ConversationMemory()
        self.loaded_at = None
        
    def load_document(self, doc_id: str, doc_text: str, chunks: List[Tuple[str, int]]):
        """
        Store loaded document and chunks.
        
        Args:
            doc_id: Document identifier
            doc_text: Full document text
            chunks: List of (chunk_text, chunk_id) tuples
        """
        self.document_id = doc_id
        self.document_text = doc_text
        self.chunks = chunks
        self.loaded_at = datetime.now()
        logger.info(f"Loaded document {doc_id} with {len(chunks)} chunks")
    
    def set_embeddings(self, vectorizer, index):
        """
        Store embeddings vectorizer and FAISS index.
        
        Args:
            vectorizer: Fitted TF-IDF vectorizer
            index: FAISS index
        """
        self.tfidf_vectorizer = vectorizer
        self.embeddings_index = index
        logger.info("Embeddings index set")
    
    def is_document_loaded(self) -> bool:
        """Check if a document is currently loaded."""
        return self.document_id is not None
    
    def clear(self):
        """Clear all session data."""
        self.document_id = None
        self.document_text = None
        self.chunks = []
        self.embeddings_index = None
        self.tfidf_vectorizer = None
        self.conversation.clear()
        self.loaded_at = None
        logger.info("Session cleared")
    
    def get_chunk_by_id(self, chunk_id: int) -> Tuple[str, int]:
        """
        Retrieve a chunk by its ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            Tuple of (chunk_text, chunk_id)
        """
        for chunk_text, cid in self.chunks:
            if cid == chunk_id:
                return chunk_text, cid
        return None, None
