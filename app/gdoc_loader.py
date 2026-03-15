"""
Google Docs loader module for RAG chatbot.
Handles fetching and parsing Google Docs with error handling.
"""

import re
import logging
from typing import Tuple
import requests

logger = logging.getLogger(__name__)


def extract_doc_id(google_docs_link: str) -> str:
    """
    Extract document ID from a Google Docs link.
    
    Supports formats:
    - https://docs.google.com/document/d/{DOC_ID}/edit
    - https://docs.google.com/document/d/{DOC_ID}/
    
    Args:
        google_docs_link: The Google Docs link
        
    Returns:
        The document ID
        
    Raises:
        ValueError: If the link format is invalid
    """
    # Pattern to match Google Docs link
    pattern = r'docs\.google\.com/document/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, google_docs_link)
    
    if not match:
        raise ValueError(
            "Invalid Google Docs link. Expected format: "
            "https://docs.google.com/document/d/{DOC_ID}/edit"
        )
    
    return match.group(1)


def fetch_google_doc(doc_id: str, timeout: int = 10) -> str:
    """
    Fetch the text content from a Google Doc.
    
    Args:
        doc_id: The Google Document ID
        timeout: Request timeout in seconds
        
    Returns:
        The document text content
        
    Raises:
        ValueError: If the document is inaccessible or private
        requests.RequestException: If the network request fails
    """
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    
    try:
        logger.info(f"Fetching Google Doc: {doc_id}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        text = response.text.strip()
        
        if not text:
            raise ValueError("The document appears to be empty or inaccessible.")
        
        logger.info(f"Successfully fetched document. Length: {len(text)} characters")
        return text
        
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out. Please try again with a shorter document.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Network connection error. Please check your internet connection.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError("Document not found. Please check the link.")
        elif e.response.status_code == 403:
            raise ValueError(
                "Document is not publicly accessible. "
                "Please ensure the sharing settings allow access."
            )
        else:
            raise ValueError(f"Failed to fetch document: HTTP {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected error fetching document: {str(e)}")
        raise ValueError(f"Failed to fetch document: {str(e)}")


def load_google_doc(google_docs_link: str) -> Tuple[str, str]:
    """
    Load a Google Doc by extracting the ID and fetching the content.
    
    Args:
        google_docs_link: The Google Docs link
        
    Returns:
        Tuple of (document_text, doc_id)
        
    Raises:
        ValueError: If link is invalid or document is inaccessible
    """
    try:
        doc_id = extract_doc_id(google_docs_link)
        doc_text = fetch_google_doc(doc_id)
        return doc_text, doc_id
    except Exception as e:
        logger.error(f"Error loading Google Doc: {str(e)}")
        raise
