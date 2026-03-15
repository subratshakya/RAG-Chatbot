/**
 * RAG Chatbot Frontend JavaScript
 * Handles UI interactions and API communication
 */

// DOM Elements
const docLinkInput = document.getElementById('docLink');
const loadBtn = document.getElementById('loadBtn');
const loadStatus = document.getElementById('loadStatus');
const docInfo = document.getElementById('docInfo');
const chunkInfo = document.getElementById('chunkInfo');

const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');

const clearBtn = document.getElementById('clearBtn');
const historyBtn = document.getElementById('historyBtn');

const historyModal = document.getElementById('historyModal');
const historyContent = document.getElementById('historyContent');
const closeModal = document.querySelector('.close');

// State
let documentLoaded = false;

// API Configuration
const API_BASE = '/api';

// Event Listeners
loadBtn.addEventListener('click', handleLoadDocument);
docLinkInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') handleLoadDocument();
});

sendBtn.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') handleQuery();
});

clearBtn.addEventListener('click', handleClearSession);
historyBtn.addEventListener('click', showHistoryModal);
closeModal.addEventListener('click', hideHistoryModal);
window.addEventListener('click', e => {
    if (e.target === historyModal) hideHistoryModal();
});

/**
 * Load a Google Doc from the provided link
 */
async function handleLoadDocument() {
    const link = docLinkInput.value.trim();
    
    if (!link) {
        showStatus('Please enter a Google Docs link', 'error');
        return;
    }
    
    loadBtn.disabled = true;
    showStatus('Loading document...', 'loading');
    
    try {
        const response = await fetch(`${API_BASE}/load-document`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ google_docs_link: link })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load document');
        }
        
        // Success
        documentLoaded = true;
        chunkInfo.textContent = `✨ ${data.chunk_count} chunks created`;
        docInfo.style.display = 'block';
        showStatus(`✅ ${data.message}`, 'success');
        
        // Clear previous messages
        clearChat();
        addSystemMessage('Document loaded! You can now ask questions about it.');
        
        // Enable query input
        queryInput.disabled = false;
        sendBtn.disabled = false;
        
    } catch (error) {
        console.error('Error loading document:', error);
        showStatus(`❌ ${error.message}`, 'error');
        documentLoaded = false;
    } finally {
        loadBtn.disabled = false;
    }
}

/**
 * Send a query to the chatbot
 */
async function handleQuery() {
    const query = queryInput.value.trim();
    
    if (!query) return;
    
    if (!documentLoaded) {
        addErrorMessage('Please load a document first');
        return;
    }
    
    // Add user message to chat
    addChatMessage(query, 'user');
    queryInput.value = '';
    sendBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to process query');
        }
        
        // Add assistant response
        addChatMessage(data.answer, 'assistant');
        
        // Log chunks used
        if (data.chunks_used && data.chunks_used.length > 0) {
            console.log('Chunks used:', data.chunks_used);
        }
        
    } catch (error) {
        console.error('Error processing query:', error);
        addErrorMessage(`Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

/**
 * Clear the current session
 */
async function handleClearSession() {
    if (!confirm('Clear session and all conversation history?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/clear-session`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to clear session');
        }
        
        // Reset UI
        documentLoaded = false;
        clearChat();
        docInfo.style.display = 'none';
        queryInput.disabled = true;
        sendBtn.disabled = true;
        queryInput.value = '';
        docLinkInput.value = '';
        loadStatus.textContent = '';
        loadStatus.className = '';
        
        addSystemMessage('Session cleared. Load a new document to continue.');
        
    } catch (error) {
        console.error('Error clearing session:', error);
        addErrorMessage(`Error: ${error.message}`);
    }
}

/**
 * Show conversation history in modal
 */
async function showHistoryModal() {
    try {
        const response = await fetch(`${API_BASE}/conversation-history`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch history');
        }
        
        const data = await response.json();
        
        if (data.turns.length === 0) {
            historyContent.innerHTML = '<p style="text-align: center; color: #6b7280;">No conversation history yet.</p>';
        } else {
            historyContent.innerHTML = data.turns.map((turn, idx) => `
                <div class="history-turn">
                    <div class="history-label">Turn ${idx + 1} - User</div>
                    <div class="history-text">${escapeHtml(turn.user)}</div>
                    <div class="history-label">Assistant</div>
                    <div class="history-text">${escapeHtml(turn.assistant)}</div>
                    ${turn.chunks && turn.chunks.length > 0 
                        ? `<div class="history-chunks">📌 Used chunks: [${turn.chunks.join(', ')}]</div>` 
                        : ''}
                </div>
            `).join('');
        }
        
        historyModal.style.display = 'flex';
    } catch (error) {
        console.error('Error fetching history:', error);
        alert('Failed to fetch conversation history');
    }
}

function hideHistoryModal() {
    historyModal.style.display = 'none';
}

/**
 * Add a message to the chat
 */
function addChatMessage(text, role = 'assistant') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;
    
    const p = document.createElement('p');
    p.innerHTML = formatMessage(text);
    msgDiv.appendChild(p);
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Add a system message
 */
function addSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message system';
    const p = document.createElement('p');
    p.textContent = text;
    msgDiv.appendChild(p);
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Add an error message
 */
function addErrorMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message error';
    const p = document.createElement('p');
    p.textContent = text;
    msgDiv.appendChild(p);
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Clear all chat messages
 */
function clearChat() {
    // Keep only the first system message
    const firstElement = Array.from(chatContainer.children).find(el => el.classList.contains('system'));
    chatContainer.innerHTML = '';
    if (firstElement) {
        chatContainer.appendChild(firstElement.cloneNode(true));
    }
}

/**
 * Format message text with citations
 */
function formatMessage(text) {
    // Escape HTML
    text = escapeHtml(text);
    
    // Format citations like [Chunk 3]
    text = text.replace(/\[Chunk (\d+)\]/g, '<span class="citation">[Chunk $1]</span>');
    
    // Add line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
    loadStatus.textContent = message;
    loadStatus.className = `status-message ${type}`;
}

/**
 * Check health and update UI on page load
 */
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.document_loaded) {
            documentLoaded = true;
            queryInput.disabled = false;
            sendBtn.disabled = false;
            docInfo.style.display = 'block';
            chunkInfo.textContent = `✨ ${data.chunk_count} chunks loaded`;
        }
    } catch (error) {
        console.error('Error checking health:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    addSystemMessage('Welcome to RAG Chatbot! Load a Google Doc to get started.');
});
