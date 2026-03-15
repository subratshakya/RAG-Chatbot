"""
Text chunking module for RAG chatbot.
Splits documents into overlapping chunks for embedding and retrieval.
"""

import logging
from typing import List, Tuple
import re

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    target_chunk_count: int = 10,
    overlap_ratio: float = 0.1,
    min_chunk_size: int = 50,
    max_chunk_size: int = 1000
) -> List[Tuple[str, int]]:
    """
    Split text into word-based overlapping chunks.
    
    Args:
        text: The full document text
        target_chunk_count: Target number of chunks (10-25 recommended)
        overlap_ratio: Overlap between chunks as a ratio (0.0-1.0)
        min_chunk_size: Minimum number of words per chunk
        max_chunk_size: Maximum number of words per chunk
        
    Returns:
        List of tuples containing (chunk_text, chunk_id)
    """
    # Clean and normalize text
    text = text.strip()
    
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
    
    # Split text into words
    words = text.split()
    
    if len(words) < min_chunk_size:
        logger.warning(f"Document has only {len(words)} words. Returning as single chunk.")
        return [(text, 0)]
    
    # Calculate optimal chunk size based on target chunk count
    optimal_chunk_size = max(len(words) // target_chunk_count, min_chunk_size)
    optimal_chunk_size = min(optimal_chunk_size, max_chunk_size)
    
    # Calculate overlap in words
    overlap_words = max(1, int(optimal_chunk_size * overlap_ratio))
    step_size = optimal_chunk_size - overlap_words
    
    logger.info(
        f"Chunking: {len(words)} words, chunk_size={optimal_chunk_size}, "
        f"overlap={overlap_words}, step_size={step_size}"
    )
    
    chunks_with_ids = []
    
    # Create overlapping chunks
    for i in range(0, len(words), step_size):
        end_idx = min(i + optimal_chunk_size, len(words))
        
        # Avoid very small final chunks
        if end_idx < len(words) and len(words) - end_idx < min_chunk_size // 2:
            end_idx = len(words)
        
        chunk_words = words[i:end_idx]
        
        if len(chunk_words) >= min_chunk_size or i == 0:
            chunk_text = " ".join(chunk_words)
            chunk_id = len(chunks_with_ids)
            chunks_with_ids.append((chunk_text, chunk_id))
        
        if end_idx == len(words):
            break
    
    logger.info(f"Created {len(chunks_with_ids)} chunks")
    return chunks_with_ids


def clean_chunk(chunk: str) -> str:
    """
    Clean a chunk by removing extra whitespace and normalizing text.
    
    Args:
        chunk: The chunk text
        
    Returns:
        Cleaned chunk text
    """
    # Remove multiple spaces
    chunk = re.sub(r'\s+', ' ', chunk)
    # Remove leading/trailing whitespace
    chunk = chunk.strip()
    return chunk


def get_chunk_preview(chunk: str, max_length: int = 100) -> str:
    """
    Get a preview of a chunk for display/logging.
    
    Args:
        chunk: The chunk text
        max_length: Maximum preview length
        
    Returns:
        Preview string
    """
    preview = chunk[:max_length]
    if len(chunk) > max_length:
        preview += "..."
    return preview
