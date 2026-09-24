import os
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma_db"

# Relevance confidence score threshold for Hallucination Guardrail (0.0 to 1.0)
RELEVANCE_THRESHOLD = 0.40


def get_embedding_function():
    """
    Returns HuggingFaceEmbeddings for local, zero-cost, high-performance RAG vector embeddings.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    use_google_embeddings = os.environ.get("USE_GOOGLE_EMBEDDINGS", "false").lower() == "true"

    if api_key and use_google_embeddings:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="text-embedding-004",
                google_api_key=api_key,
            )
        except Exception as e:
            print(f"Warning: Failed to load GoogleGenerativeAIEmbeddings ({e}). Falling back to local HuggingFace embeddings.")

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class RAGManager:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR, persist_dir: Path = CHROMA_PERSIST_DIR):
        self.knowledge_dir = knowledge_dir
        self.persist_dir = persist_dir
        self.embedding_fn = get_embedding_function()
        self.vector_store = None
        self.total_docs = 0
        self.total_chunks = 0
        self.initialize_vector_store()

    def load_knowledge_documents(self) -> List[Document]:
        """Loads all text and markdown knowledge documents from knowledge directory."""
        documents = []
        if not self.knowledge_dir.exists():
            print(f"Knowledge directory '{self.knowledge_dir}' does not exist.")
            return documents

        for filepath in self.knowledge_dir.glob("*.*"):
            if filepath.suffix.lower() in [".txt", ".md"]:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        doc = Document(
                            page_content=text,
                            metadata={"source": filepath.name, "path": str(filepath)}
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error reading file '{filepath}': {e}")

        self.total_docs = len(documents)
        return documents

    def initialize_vector_store(self, force_reindex: bool = False):
        """Loads documents, splits into chunks, and persists embeddings in ChromaDB."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        docs = self.load_knowledge_documents()
        if not docs:
            print("No knowledge documents found to index.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        self.total_chunks = len(chunks)

        print(f"Indexing RAG Knowledge Base: {self.total_docs} documents split into {self.total_chunks} chunks.")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn,
            persist_directory=str(self.persist_dir),
            collection_name="anavi_knowledge_base"
        )
        print("ChromaDB persistent vector store initialized successfully.")

    def query_with_guardrail(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top-k relevant context chunks from ChromaDB.
        Applies similarity score thresholding for Hallucination Protection.
        If confidence < RELEVANCE_THRESHOLD, returns guarded rejection.
        """
        if not self.vector_store:
            return {
                "success": False,
                "reason": "vectorstore_uninitialized",
                "message": "Knowledge base vector store is not initialized.",
                "context": "",
                "best_score": 0.0,
                "chunks_retrieved": 0,
            }

        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)

        if not results_with_scores:
            return {
                "success": False,
                "reason": "no_results",
                "message": "I don't have enough information in my knowledge base to answer that.",
                "context": "",
                "best_score": 0.0,
                "chunks_retrieved": 0,
            }

        processed_results = []
        best_score = 0.0

        for doc, distance in results_with_scores:
            similarity = 1.0 / (1.0 + float(distance))
            best_score = max(best_score, similarity)
            processed_results.append((doc, similarity))

        print(f"RAG Query: '{query}' -> Top Similarity Score: {best_score:.4f} (Threshold: {RELEVANCE_THRESHOLD})")

        # Hallucination Guardrail Check
        if best_score < RELEVANCE_THRESHOLD:
            return {
                "success": False,
                "reason": "insufficient_relevance",
                "message": "I don't have enough information in my knowledge base to answer that.",
                "context": "",
                "best_score": round(best_score, 4),
                "chunks_retrieved": len(results_with_scores),
            }

        relevant_chunks = [
            f"[Source: {doc.metadata.get('source', 'Knowledge Base')}]\n{doc.page_content}"
            for doc, score in processed_results
        ]

        context_str = "\n\n---\n\n".join(relevant_chunks)

        return {
            "success": True,
            "message": "Retrieved relevant context from knowledge base.",
            "context": context_str,
            "best_score": round(best_score, 4),
            "chunks_retrieved": len(relevant_chunks),
        }


# Singleton instance
rag_instance = RAGManager()
