import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Search, Trash2, Settings, MessageSquare, Database, Cpu, AlertCircle, CheckCircle, Loader, ChevronDown, ChevronUp } from 'lucide-react';

const IndustrialRAGSystem = () => {
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [vectorStore, setVectorStore] = useState([]);
  const [settings, setSettings] = useState({
    chunkSize: 500,
    chunkOverlap: 50,
    topK: 3,
    temperature: 0.7,
    model: 'meta-llama/llama-3.2-3b-instruct'
  });
  const [showSettings, setShowSettings] = useState(false);
  const [stats, setStats] = useState({
    totalDocs: 0,
    totalChunks: 0,
    avgChunkSize: 0
  });
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Text chunking with overlap for better context preservation
  const chunkText = (text, chunkSize, overlap) => {
    const chunks = [];
    let start = 0;
    
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      chunks.push(text.slice(start, end));
      start += chunkSize - overlap;
    }
    
    return chunks;
  };

  // Simple embedding simulation (in production, use actual embedding models)
  const generateEmbedding = (text) => {
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(384).fill(0);
    
    words.forEach((word, idx) => {
      for (let i = 0; i < word.length; i++) {
        const charCode = word.charCodeAt(i);
        embedding[(idx + i * 7) % 384] += charCode / 1000;
      }
    });
    
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / magnitude);
  };

  // Cosine similarity for vector comparison
  const cosineSimilarity = (vec1, vec2) => {
    return vec1.reduce((sum, val, idx) => sum + val * vec2[idx], 0);
  };

  // Handle file upload and processing
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    setIsProcessing(true);

    for (const file of files) {
      const text = await file.text();
      const chunks = chunkText(text, settings.chunkSize, settings.chunkOverlap);
      
      const newDoc = {
        id: Date.now() + Math.random(),
        name: file.name,
        size: file.size,
        uploadedAt: new Date().toISOString(),
        content: text,
        chunks: chunks.length
      };

      setDocuments(prev => [...prev, newDoc]);

      // Generate embeddings for each chunk
      const newVectors = chunks.map((chunk, idx) => ({
        id: `${newDoc.id}_chunk_${idx}`,
        docId: newDoc.id,
        docName: file.name,
        text: chunk,
        embedding: generateEmbedding(chunk),
        chunkIndex: idx
      }));

      setVectorStore(prev => [...prev, ...newVectors]);
    }

    updateStats();
    setIsProcessing(false);
    addSystemMessage(`Successfully processed ${files.length} document(s)`);
  };

  // Update statistics
  const updateStats = () => {
    setStats({
      totalDocs: documents.length + 1,
      totalChunks: vectorStore.length,
      avgChunkSize: vectorStore.length > 0 
        ? Math.round(vectorStore.reduce((sum, v) => sum + v.text.length, 0) / vectorStore.length)
        : 0
    });
  };

  // Retrieve relevant chunks using vector similarity
  const retrieveRelevantChunks = (query, topK) => {
    const queryEmbedding = generateEmbedding(query);
    
    const scoredChunks = vectorStore.map(vector => ({
      ...vector,
      score: cosineSimilarity(queryEmbedding, vector.embedding)
    }));

    scoredChunks.sort((a, b) => b.score - a.score);
    return scoredChunks.slice(0, topK);
  };

  // Add system message
  const addSystemMessage = (content) => {
    setMessages(prev => [...prev, {
      role: 'system',
      content,
      timestamp: new Date().toISOString()
    }]);
  };

  // Handle RAG query with LLM
  const handleQuery = async () => {
    if (!query.trim() || vectorStore.length === 0) {
      addSystemMessage('Please upload documents and enter a query');
      return;
    }

    setIsProcessing(true);
    
    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString()
    }]);

    try {
      // Retrieve relevant chunks
      const relevantChunks = retrieveRelevantChunks(query, settings.topK);
      
      // Build context from retrieved chunks
      const context = relevantChunks
        .map((chunk, idx) => `[Document: ${chunk.docName}, Chunk ${chunk.chunkIndex + 1}]\n${chunk.text}`)
        .join('\n\n---\n\n');

      // Build prompt for LLM
      const prompt = `You are a helpful AI assistant. Answer the user's question based on the provided context. If the context doesn't contain relevant information, say so clearly.

Context from documents:
${context}

User Question: ${query}

Please provide a comprehensive answer based on the context above. Cite specific documents when relevant.`;

      // Call OpenRouter API
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_OPENROUTER_API_KEY}`,
        },
        body: JSON.stringify({
          model: settings.model,
          messages: [
            { role: 'user', content: prompt }
          ],
          temperature: settings.temperature,
          max_tokens: 1000
        })
      });

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error.message);
      }
      const assistantResponse = data.choices[0].message.content;

      // Add assistant message with sources
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: assistantResponse,
        sources: relevantChunks.map(c => ({
          doc: c.docName,
          chunk: c.chunkIndex + 1,
          score: c.score.toFixed(3)
        })),
        timestamp: new Date().toISOString()
      }]);

    } catch (error) {
      addSystemMessage(`Error: ${error.message}`);
    }

    setQuery('');
    setIsProcessing(false);
  };

  // Delete document
  const deleteDocument = (docId) => {
    setDocuments(prev => prev.filter(doc => doc.id !== docId));
    setVectorStore(prev => prev.filter(vec => vec.docId !== docId));
    updateStats();
    addSystemMessage('Document deleted successfully');
  };

  // Clear all data
  const clearAll = () => {
    setDocuments([]);
    setVectorStore([]);
    setMessages([]);
    setStats({ totalDocs: 0, totalChunks: 0, avgChunkSize: 0 });
    addSystemMessage('All data cleared');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Industrial RAG System
          </h1>
          <p className="text-slate-300">Retrieval-Augmented Generation with Claude Integration</p>
        </div>

        {/* Statistics Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-slate-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">Total Documents</p>
                <p className="text-2xl font-bold text-blue-400">{stats.totalDocs}</p>
              </div>
              <Database className="w-8 h-8 text-blue-400" />
            </div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-slate-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">Vector Chunks</p>
                <p className="text-2xl font-bold text-cyan-400">{stats.totalChunks}</p>
              </div>
              <Cpu className="w-8 h-8 text-cyan-400" />
            </div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-slate-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">Avg Chunk Size</p>
                <p className="text-2xl font-bold text-green-400">{stats.avgChunkSize}</p>
              </div>
              <FileText className="w-8 h-8 text-green-400" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Document Management */}
          <div className="lg:col-span-1 space-y-4">
            {/* Upload Section */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Documents
              </h2>
              <label className="block">
                <input
                  type="file"
                  multiple
                  accept=".txt,.md,.csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={isProcessing}
                />
                <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors">
                  <Upload className="w-12 h-12 mx-auto mb-2 text-slate-400" />
                  <p className="text-sm text-slate-400">Click to upload or drag files</p>
                  <p className="text-xs text-slate-500 mt-1">TXT, MD, CSV</p>
                </div>
              </label>
            </div>

            {/* Documents List */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700 max-h-96 overflow-y-auto">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Documents ({documents.length})
              </h2>
              {documents.length === 0 ? (
                <p className="text-slate-400 text-sm text-center py-8">No documents uploaded</p>
              ) : (
                <div className="space-y-2">
                  {documents.map(doc => (
                    <div key={doc.id} className="bg-slate-700/50 rounded p-3 flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{doc.name}</p>
                        <p className="text-xs text-slate-400">{doc.chunks} chunks • {(doc.size / 1024).toFixed(1)} KB</p>
                      </div>
                      <button
                        onClick={() => deleteDocument(doc.id)}
                        className="ml-2 p-1 hover:bg-red-500/20 rounded transition-colors"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {documents.length > 0 && (
                <button
                  onClick={clearAll}
                  className="mt-4 w-full bg-red-500/20 text-red-400 py-2 rounded hover:bg-red-500/30 transition-colors text-sm"
                >
                  Clear All Documents
                </button>
              )}
            </div>

            {/* Settings */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="w-full flex items-center justify-between mb-4"
              >
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  Settings
                </h2>
                {showSettings ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              {showSettings && (
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">Chunk Size</label>
                    <input
                      type="number"
                      value={settings.chunkSize}
                      onChange={(e) => setSettings({...settings, chunkSize: parseInt(e.target.value)})}
                      className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">Chunk Overlap</label>
                    <input
                      type="number"
                      value={settings.chunkOverlap}
                      onChange={(e) => setSettings({...settings, chunkOverlap: parseInt(e.target.value)})}
                      className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">Top K Results</label>
                    <input
                      type="number"
                      value={settings.topK}
                      onChange={(e) => setSettings({...settings, topK: parseInt(e.target.value)})}
                      className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">Temperature</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={settings.temperature}
                      onChange={(e) => setSettings({...settings, temperature: parseFloat(e.target.value)})}
                      className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Chat Interface */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 h-[calc(100vh-280px)] flex flex-col">
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-700">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <MessageSquare className="w-5 h-5" />
                  RAG Chat Interface
                </h2>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center py-12 text-slate-400">
                    <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>Upload documents and start asking questions</p>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-3xl rounded-lg p-4 ${
                        msg.role === 'user' 
                          ? 'bg-blue-600 text-white' 
                          : msg.role === 'system'
                          ? 'bg-yellow-600/20 text-yellow-300 border border-yellow-600/30'
                          : 'bg-slate-700 text-white'
                      }`}>
                        {msg.role === 'system' && (
                          <div className="flex items-center gap-2 mb-2">
                            <AlertCircle className="w-4 h-4" />
                            <span className="text-xs font-semibold">SYSTEM</span>
                          </div>
                        )}
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        {msg.sources && (
                          <div className="mt-3 pt-3 border-t border-slate-600">
                            <p className="text-xs font-semibold mb-2 text-slate-300">Sources:</p>
                            {msg.sources.map((source, sidx) => (
                              <div key={sidx} className="text-xs text-slate-400 mb-1">
                                📄 {source.doc} (Chunk {source.chunk}) - Similarity: {source.score}
                              </div>
                            ))}
                          </div>
                        )}
                        <p className="text-xs text-slate-400 mt-2">{new Date(msg.timestamp).toLocaleTimeString()}</p>
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-slate-700">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && !isProcessing && handleQuery()}
                    placeholder="Ask a question about your documents..."
                    className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500"
                    disabled={isProcessing || vectorStore.length === 0}
                  />
                  <button
                    onClick={handleQuery}
                    disabled={isProcessing || !query.trim() || vectorStore.length === 0}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
                  >
                    {isProcessing ? (
                      <>
                        <Loader className="w-5 h-5 animate-spin" />
                        Processing
                      </>
                    ) : (
                      <>
                        <Search className="w-5 h-5" />
                        Search
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IndustrialRAGSystem;