"""
RAG (Retrieval-Augmented Generation) module for the chatbot.
Handles embeddings, retrieval, and LLM integration with Groq.
"""

import logging
import os
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import requests

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine using TF-IDF and FAISS.
    """
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize RAG engine.
        
        Args:
            groq_api_key: API key for Groq LLM
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.vectorizer = None
        self.faiss_index = None
        self.chunks = []
        self.model_name = "llama-3.1-8b-instant"
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set. LLM generation will fail.")
    
    def create_embeddings(self, chunks: List[Tuple[str, int]]) -> Tuple:
        """
        Create TF-IDF embeddings for chunks and build FAISS index.
        
        Args:
            chunks: List of (chunk_text, chunk_id) tuples
            
        Returns:
            Tuple of (vectorizer, faiss_index)
        """
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        
        self.chunks = chunks
        chunk_texts = [text for text, _ in chunks]
        
        # Dynamically adjust parameters based on number of chunks
        num_chunks = len(chunk_texts)
        
        # For very small datasets, use conservative settings
        if num_chunks <= 5:
            min_df_val = 1
            max_df_val = num_chunks  # Allow all documents
            ngram_range = (1, 1)  # Only unigrams for small datasets
        elif num_chunks <= 20:
            min_df_val = 1
            max_df_val = max(2, num_chunks - 1)  # Leave at least 1 doc
            ngram_range = (1, 2)
        else:
            min_df_val = max(1, num_chunks // 20)
            max_df_val = max(0.9, 1.0 - (2.0 / num_chunks))
            ngram_range = (1, 2)
        
        logger.info(f"TF-IDF parameters: chunks={num_chunks}, min_df={min_df_val}, max_df={max_df_val}, ngrams={ngram_range}")
        
        # Create TF-IDF vectorizer with safe parameters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=ngram_range,
            min_df=min_df_val,
            max_df=max_df_val,
            lowercase=True,
            stop_words='english'
        )
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        
        # Convert to dense array for FAISS
        tfidf_dense = tfidf_matrix.toarray().astype('float32')
        
        # Create FAISS index
        dimension = tfidf_dense.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(tfidf_dense)
        
        logger.info(f"Created FAISS index with dimension {dimension}")
        return self.vectorizer, self.faiss_index
    
    def retrieve_chunks(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[str, int, float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User query
            k: Number of chunks to retrieve (top-k)
            score_threshold: Minimum similarity score (0.0 = no minimum)
            
        Returns:
            List of (chunk_text, chunk_id, similarity_score) tuples
            
        Raises:
            ValueError: If index is not initialized
        """
        if self.faiss_index is None or self.vectorizer is None:
            raise ValueError("Embeddings not created. Load a document first.")
        
        # Transform query using same vectorizer
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_vector, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # FAISS returns L2 distance, convert to similarity
            similarity = 1 / (1 + distance)
            
            if similarity >= score_threshold:
                chunk_text, chunk_id = self.chunks[idx]
                results.append((chunk_text, chunk_id, float(similarity)))
        
        logger.debug(f"Retrieved {len(results)} chunks for query")
        return results
    
    def format_context(self, retrieved_chunks: List[Tuple[str, int, float]]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            retrieved_chunks: List of (chunk_text, chunk_id, similarity) tuples
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant information found in the document."
        
        context_lines = ["## Relevant Document Excerpts:\n"]
        for chunk_text, chunk_id, similarity in retrieved_chunks:
            context_lines.append(f"[Chunk {chunk_id}] (similarity: {similarity:.3f})")
            context_lines.append(chunk_text)
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: str = "",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate response using Groq LLM with retrieval context.
        
        Args:
            query: User's question
            context: Retrieved context from chunks
            conversation_history: Previous conversation turns
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum response tokens
            
        Returns:
            Generated response with citations
            
        Raises:
            ValueError: If API key is missing or API call fails
        """
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not configured")
        
        # Build system prompt
        system_prompt = """You are a helpful AI assistant answering questions about a document.
Answer based ONLY on the provided document excerpts. 
Always cite the specific chunks you use like [Chunk X].
If information is not in the document, say "This information is not available in the document."
Keep responses concise and focused."""
        
        # Build conversation context
        history_part = f"{conversation_history}\n\n" if conversation_history else ""
        
        # Build user message with context
        user_message = f"""{history_part}{context}

User Question: {query}

Please answer the question above using only the provided document excerpts. Remember to cite chunks like [Chunk X]."""
        
        try:
            logger.info("Calling Groq API for response generation")
            
            response = requests.post(
                self.groq_api_url,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.95
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result["choices"][0]["message"]["content"]
            logger.info("Successfully generated response from Groq")
            
            return generated_text.strip()
            
        except requests.exceptions.Timeout:
            return "Error: Request to LLM timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "Error: Failed to connect to LLM service. Please check your connection."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "Error: Invalid API key. Please check your GROQ_API_KEY."
            elif e.response.status_code == 429:
                return "Error: Rate limited by LLM service. Please try again later."
            else:
                logger.error(f"LLM API error: {e}")
                return f"Error: Failed to generate response (HTTP {e.response.status_code})"
        except Exception as e:
            logger.error(f"Unexpected error generating response: {str(e)}")
            return f"Error: Failed to generate response. {str(e)}"


def extract_chunk_ids_from_response(response_text: str) -> List[int]:
    """
    Extract chunk IDs from response text (format: [Chunk X]).
    
    Args:
        response_text: Generated response text
        
    Returns:
        List of chunk IDs referenced in response
    """
    import re
    pattern = r'\[Chunk (\d+)\]'
    matches = re.findall(pattern, response_text)
    return [int(m) for m in matches]
