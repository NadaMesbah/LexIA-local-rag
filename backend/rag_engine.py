"""
RAG Engine — Core logic
Connects LlamaIndex orchestration + Milvus vector store + Ollama local LLM.

Architecture:
  - Milvus stores vectors PERSISTENTLY (survives restarts)
  - On startup: connect to existing collection (no re-indexing)
  - On upload: incrementally add new chunks to existing collection
  - Ollama serves the local LLM (Mistral / LLaMA 3 / Qwen)
"""

import os
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor


# ─── Configuration ────────────────────────────────────────────────────────────

MILVUS_URI       = os.getenv("MILVUS_URI", "http://localhost:19530")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL", "qwen2.5")          # or llama3, qwen2.5
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "rag_knowledge_base")
EMBED_DIM        = int(os.getenv("EMBED_DIM", "768"))

# Chunking strategy — sentence-aware splitting
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "64"))


class RAGEngine:
    """
    Persistent RAG engine.
    The knowledge base lives in Milvus — no data lost on restart.
    """

    def __init__(self):
        self._index: Optional[VectorStoreIndex] = None
        self._vector_store: Optional[MilvusVectorStore] = None
        self._doc_registry: dict[str, int] = {}   # filename → chunk count
        self._setup()

    # ─── Setup ────────────────────────────────────────────────────────────────

    def _setup(self):
        """Initialize models and connect to Milvus collection."""
        print(f"[RAG] Connecting to Ollama at {OLLAMA_BASE_URL}...")

        # Local LLM via Ollama
        llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=600.0,   
            context_window=8192,
        )

        # Local embedding model via Ollama
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            ollama_additional_kwargs={"mirostat": 0},
        )

        # Set globally so all LlamaIndex components use them
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        Settings.num_output = 1024
        Settings.context_window = 8192

        print(f"[RAG] Connecting to Milvus at {MILVUS_URI}...")

        # Connect to Milvus — collection persists between restarts
        self._vector_store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=COLLECTION_NAME,
            dim=EMBED_DIM,
            overwrite=False,   # ← KEY: never wipe existing data on startup
        )

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        # Load existing index from Milvus (no documents needed)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            storage_context=storage_context,
        )

        print(f"[RAG] ✓ Engine ready. Collection: '{COLLECTION_NAME}'")

    # ─── Query ────────────────────────────────────────────────────────────────

    async def query(self, question: str, top_k: int = 5) -> dict:
        """Query the knowledge base and return answer + source chunks."""
        loop = asyncio.get_event_loop()

        def _run():
            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
            )
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)],
            )
            response = query_engine.query(question)

            sources = []
            for node in response.source_nodes:
                sources.append({
                    "text":       node.node.get_content()[:300] + "...",
                    "score":      round(float(node.score or 0), 4),
                    "filename":   node.node.metadata.get("file_name", "Unknown"),
                    "page":       node.node.metadata.get("page_label", "?"),
                })

            return {
                "answer":  str(response),
                "sources": sources,
            }

        return await loop.run_in_executor(None, _run)

    async def query_stream(self, question: str, top_k: int = 5) -> AsyncGenerator[str, None]:
        """Streaming query — yields tokens as they arrive from Ollama."""
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=top_k,
        )
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            streaming=True,
        )
        response = query_engine.query(question)
        for token in response.response_gen:
            yield token

    # ─── Ingestion ────────────────────────────────────────────────────────────

    async def ingest_document(self, file_path: str, original_name: str) -> dict:
        """Add a single document to the persistent knowledge base."""
        loop = asyncio.get_event_loop()

        def _run():
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()

            # Tag every node with the original filename
            for doc in documents:
                doc.metadata["file_name"] = original_name
                doc.metadata["source"] = original_name

            # Incremental insert — adds to existing collection
            for doc in documents:
                self._index.insert(doc)

            # Track in registry
            self._doc_registry[original_name] = len(documents)
            return {"chunks": len(documents), "filename": original_name}

        return await loop.run_in_executor(None, _run)

    async def ingest_folder(self, folder_path: str) -> dict:
        """Bulk-ingest all supported files from a folder."""
        folder = Path(folder_path)
        if not folder.exists():
            return {"error": f"Folder '{folder_path}' does not exist."}

        supported = [".pdf", ".txt", ".docx", ".md", ".csv"]
        files = [f for f in folder.rglob("*") if f.suffix.lower() in supported]

        if not files:
            return {"message": "No supported documents found.", "ingested": 0}

        total_chunks = 0
        ingested_files = []

        for file in files:
            result = await self.ingest_document(str(file), file.name)
            total_chunks += result["chunks"]
            ingested_files.append(file.name)

        return {
            "message":    f"Ingested {len(ingested_files)} documents.",
            "files":      ingested_files,
            "total_chunks": total_chunks,
        }

    # ─── Management ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Check if the knowledge base has any content."""
        try:
            count = self._vector_store._milvus_client.num_entities(COLLECTION_NAME)
            return count > 0
        except Exception:
            return len(self._doc_registry) > 0

    def get_status(self) -> dict:
        """Return knowledge base stats."""
        try:
            count = self._vector_store._milvus_client.num_entities(COLLECTION_NAME)
        except Exception:
            count = -1
        return {
            "status":          "ready",
            "doc_count":       count,
            "collection_name": COLLECTION_NAME,
        }

    def list_documents(self) -> dict:
        """List tracked documents."""
        return {
            "documents": [
                {"name": name, "chunks": chunks}
                for name, chunks in self._doc_registry.items()
            ]
        }

    def delete_document(self, doc_name: str) -> bool:
        """Remove a document's chunks from Milvus by metadata filter."""
        if doc_name not in self._doc_registry:
            return False
        try:
            self._vector_store._milvus_client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'metadata["file_name"] == "{doc_name}"',
            )
            del self._doc_registry[doc_name]
            return True
        except Exception as e:
            print(f"[RAG] Delete error: {e}")
            return False

    def reset(self):
        """Wipe the entire collection and re-create it empty."""
        self._vector_store._milvus_client.drop_collection(COLLECTION_NAME)
        self._doc_registry.clear()
        self._setup()
        print("[RAG] Knowledge base reset.")