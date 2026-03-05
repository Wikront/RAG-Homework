"""
RAG Chatbot — source package.

Modules:
    loaders       — ingest documents (PDFLoader, AudioLoader)
    chunking      — split text into overlapping chunks (TextChunker, Chunk)
    embeddings    — encode text into dense vectors (EmbeddingModel)
    vector_store  — persist and query a ChromaDB collection (ChromaStore, SearchResult)
    rag           — retrieval-augmented generation pipeline (RAGPipeline, OpenAIClient, RAGResponse)
    chat          — interactive CLI (run via `python -m src.chat`)
"""
