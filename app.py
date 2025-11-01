"""
Professional RAG System with IVF Clustering
==========================================
Features:
- Text chunking with metadata (page numbers, char positions)
- OpenAI embeddings with optimal model selection
- FAISS IVF clustering for efficient similarity search
- Out-of-context query detection with optimal thresholds
- Professional prompt engineering for accurate responses
"""

import os
import time
import hashlib
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for each text chunk"""
    chunk_id: str
    source_page: int
    char_start: int
    char_end: int
    document_name: str
    chunk_size: int

@dataclass
class QueryResult:
    """Result structure for query processing"""
    query: str
    response: str
    retrieved_chunks: List[Dict]
    similarity_scores: List[float]
    processing_time: float
    within_context: bool
    chunk_ids: List[str]

class DocumentProcessor:
    """Handles PDF processing and text chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process PDF and return chunks with metadata"""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            document_name = Path(pdf_path).stem
            
            logger.info(f"Processing {len(documents)} pages from {document_name}")
            
            all_chunks = []
            global_char_position = 0
            
            for page_idx, doc in enumerate(documents):
                page_content = doc.page_content
                page_number = page_idx + 1
                
                # Split page content into chunks
                chunks = self.text_splitter.split_text(page_content)
                
                page_char_start = global_char_position
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    # Generate unique chunk ID
                    chunk_id = self._generate_chunk_id(document_name, page_number, chunk_idx)
                    
                    # Calculate character positions
                    char_start = global_char_position
                    char_end = global_char_position + len(chunk_text)
                    
                    # Create metadata
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        source_page=page_number,
                        char_start=char_start,
                        char_end=char_end,
                        document_name=document_name,
                        chunk_size=len(chunk_text)
                    )
                    
                    chunk_data = {
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'metadata': metadata
                    }
                    
                    all_chunks.append(chunk_data)
                    
                    # Update position for next chunk (considering overlap)
                    global_char_position = char_end - self.chunk_overlap if chunk_idx < len(chunks) - 1 else char_end
                
                global_char_position = char_end  # Reset for next page
            
            logger.info(f"Created {len(all_chunks)} chunks from {document_name}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _generate_chunk_id(self, doc_name: str, page_num: int, chunk_idx: int) -> str:
        """Generate unique chunk ID"""
        return f"{doc_name}_page_{page_num:03d}_chunk_{chunk_idx:03d}"

class OptimalEmbeddingManager:
    """Manages OpenAI embeddings with optimal configuration"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model,
            chunk_size=1000  # Optimal batch size for API
        )
        
        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-small": {"dimension": 1536, "optimal_threshold": 0.45},
            "text-embedding-3-large": {"dimension": 3072, "optimal_threshold": 0.35},
            "text-embedding-ada-002": {"dimension": 1536, "optimal_threshold": 0.82}
        }
        
        self.dimension = self.model_configs[model]["dimension"]
        self.optimal_threshold = self.model_configs[model]["optimal_threshold"]
        
        logger.info(f"Initialized {model} with dimension {self.dimension}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            logger.info(f"Generated {embeddings_array.shape[0]} embeddings")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for single query"""
        try:
            embedding = self.embeddings.embed_query(query)
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            
            return embedding_array
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

class OptimalFAISSVectorStore:
    """FAISS vector store with IVF clustering optimization"""
    
    def __init__(self, dimension: int, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = None
        self.chunk_id_mapping = {}  # Maps FAISS index -> chunk_id
        self.metadata_store = {}    # Maps chunk_id -> metadata
        self.is_trained = False
        
    def _calculate_optimal_clusters(self, n_vectors: int) -> int:
        """Calculate optimal number of clusters for IVF"""
        if n_vectors < 100:
            return min(n_vectors // 2, 8)
        elif n_vectors < 1000:
            return int(4 * np.sqrt(n_vectors))
        else:
            return int(8 * np.sqrt(n_vectors))
    
    def _calculate_optimal_nprobe(self, n_clusters: int) -> int:
        """Calculate optimal nprobe value"""
        return max(1, min(n_clusters // 4, 64))  # 25% of clusters, max 64
    
    def build_index(self, embeddings: np.ndarray, chunk_data: List[Dict]):
        """Build IVF index with optimal clustering"""
        n_vectors = embeddings.shape[0]
        n_clusters = self._calculate_optimal_clusters(n_vectors)
        
        logger.info(f"Building IVF index: {n_vectors} vectors, {n_clusters} clusters")
        
        # Create IVF index
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
        
        # Use GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU acceleration")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Train the index
        logger.info("Training IVF index...")
        self.index.train(embeddings)
        self.is_trained = True
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Set optimal nprobe
        nprobe = self._calculate_optimal_nprobe(n_clusters)
        self.index.nprobe = nprobe
        # Store mappings
        for i, chunk in enumerate(chunk_data):
            chunk_id = chunk['chunk_id']
            self.chunk_id_mapping[i] = chunk_id
            self.metadata_store[chunk_id] = chunk['metadata']
        
        logger.info(f"Index built successfully: nprobe={nprobe}, clusters={n_clusters}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.45) -> Tuple[List[str], List[float]]:
        """Search for similar chunks with optimal threshold"""
        if not self.is_trained:
            raise ValueError("Index not trained. Call build_index first.")
        
        # Ensure query is normalized and correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search with extra candidates to account for threshold filtering
        search_k = min(k * 3, len(self.chunk_id_mapping))
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Filter by threshold and get chunk IDs
        filtered_chunk_ids = []
        filtered_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:  # Valid index and above threshold
                chunk_id = self.chunk_id_mapping[idx]
                filtered_chunk_ids.append(chunk_id)
                filtered_scores.append(float(score))
        
        # Limit to k results
        return filtered_chunk_ids[:k], filtered_scores[:k]

class ProfessionalRAGSystem:
    """Complete RAG system with professional prompt engineering"""
    
    def __init__(self, openai_api_key: str, embedding_model: str = "text-embedding-3-small"):
        self.api_key = openai_api_key
        self.embedding_manager = OptimalEmbeddingManager(openai_api_key, embedding_model)
        self.vector_store = OptimalFAISSVectorStore(self.embedding_manager.dimension)
        self.chunk_store = {}  # Maps chunk_id -> chunk_text
        
        # Initialize ChatGPT with optimal settings
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o-mini",  # Optimal model for RAG
            temperature=0.1,      # Low temperature for factual responses
            max_tokens=1000       # Adequate for detailed responses
        )
        
        self.professional_prompt = """You are a highly skilled AI research assistant with expertise in analyzing and synthesizing information from documents. Your role is to provide accurate, comprehensive, and well-structured responses based solely on the provided context.

INSTRUCTIONS:
1. Analyze the provided context carefully and thoroughly
2. Answer the question using ONLY information from the given context
3. Provide specific details, examples, and evidence from the context when available
4. If the context contains page references, include them in your response
5. Structure your response logically with clear reasoning
6. If the context is insufficient to fully answer the question, clearly state what information is missing
7. Do not make assumptions or add information not present in the context
8. Maintain a professional, authoritative tone throughout your response

CONTEXT:
{context}

QUESTION: {question}

RESPONSE:"""
    
    def ingest_documents(self, pdf_paths: List[str]):
        """Ingest PDF documents and build vector index"""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        
        # Process all PDFs
        for pdf_path in pdf_paths:
            chunks = processor.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No chunks extracted from documents")
        
        # Extract texts and store chunk mappings
        texts = []
        for chunk in all_chunks:
            chunk_id = chunk['chunk_id']
            text = chunk['text']
            self.chunk_store[chunk_id] = text
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        # Build vector index
        self.vector_store.build_index(embeddings, all_chunks)
        
        logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(pdf_paths)} documents")
    
    def query(self, question: str, k: int = 5) -> QueryResult:
        """Process query and return comprehensive results"""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_query_embedding(question)
        
        # Search for relevant chunks
        threshold = self.embedding_manager.optimal_threshold
        chunk_ids, scores = self.vector_store.search(query_embedding, k=k, threshold=threshold)
    
        # Check if query is within context
        within_context = len(chunk_ids) > 0 and len(scores) > 0 and max(scores) >= threshold
        
        if not within_context:
            response = """I apologize, but your query appears to be outside the scope of the available documents. The similarity scores for your question are below the confidence threshold, indicating that the documents may not contain relevant information to provide an accurate response.

Please try:
1. Rephrasing your question to better match the document content
2. Asking about topics that are directly covered in the uploaded documents
3. Being more specific about the particular aspect you're interested in"""
            
            return QueryResult(
                query=question,
                response=response,
                retrieved_chunks=[],
                similarity_scores=[],
                processing_time=time.time() - start_time,
                within_context=False,
                chunk_ids=[]
            )
        
        # Retrieve chunk contents and metadata
        retrieved_chunks = []
        for chunk_id in chunk_ids:
            chunk_text = self.chunk_store[chunk_id]
            metadata = self.vector_store.metadata_store[chunk_id]
            
            retrieved_chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'metadata': metadata
            })
        
        # Prepare context for LLM
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk['metadata']
            context_part = f"""[Document: {metadata.document_name}, Page: {metadata.source_page}, Relevance: {scores[i]:.3f}]
{chunk['text']}"""
            context_parts.append(context_part)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response using professional prompt
        prompt = self.professional_prompt.format(context=context, question=question)
        
        try:
            response = self.llm.predict(prompt)
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            response = "I encountered an error while processing your question. Please try again."
        
        processing_time = time.time() - start_time
        
        return QueryResult(
            query=question,
            response=response,
            retrieved_chunks=retrieved_chunks,
            similarity_scores=scores,
            processing_time=processing_time,
            within_context=True,
            chunk_ids=chunk_ids
        )
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_chunks': len(self.chunk_store),
            'embedding_model': self.embedding_manager.model,
            'embedding_dimension': self.embedding_manager.dimension,
            'optimal_threshold': self.embedding_manager.optimal_threshold,
            'index_trained': self.vector_store.is_trained,
            'nprobe': getattr(self.vector_store.index, 'nprobe', 'N/A') if self.vector_store.index else 'N/A'
        }

def main():
    """Example usage of the Professional RAG System"""
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize RAG system
    rag_system = ProfessionalRAGSystem(
        openai_api_key=OPENAI_API_KEY,
        embedding_model="text-embedding-3-small"  # Optimal for most use cases
    )
    
    # Example document paths (replace with your PDF paths)
    pdf_paths = [
        "document1.pdf",
        "document2.pdf"
    ]
    
    try:
        # Ingest documents
        print("🔄 Ingesting documents...")
        rag_system.ingest_documents(pdf_paths)
        
        # Display system stats
        stats = rag_system.get_system_stats()
        print(f"\n📊 System Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example queries
        queries = [
            "What are the main findings discussed in the documents?",
            "Can you summarize the key methodologies mentioned?",
            "What recommendations are provided?",
            "Tell me about quantum computing"  # Likely out of context
        ]
        
        print(f"\n🔍 Processing queries with optimal threshold: {stats['optimal_threshold']}")
        
        for query in queries:
            print(f"\n" + "="*80)
            print(f"Query: {query}")
            print("="*80)
            
            result = rag_system.query(query, k=3)
            
            print(f"⚡ Processing time: {result.processing_time:.3f}s")
            print(f"🎯 Within context: {'Yes' if result.within_context else 'No'}")
            print(f"📄 Retrieved chunks: {len(result.chunk_ids)}")
            
            if result.similarity_scores:
                print(f"📊 Similarity scores: {[f'{score:.3f}' for score in result.similarity_scores]}")
            
            print(f"\n🤖 Response:")
            print(result.response)
            
            if result.chunk_ids:
                print(f"\n📚 Source chunks: {', '.join(result.chunk_ids)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()
