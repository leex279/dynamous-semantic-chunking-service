import os
import gc
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))
BREAKPOINT_THRESHOLD_TYPE = os.getenv("BREAKPOINT_THRESHOLD_TYPE", "percentile")
BREAKPOINT_THRESHOLD_AMOUNT = float(os.getenv("BREAKPOINT_THRESHOLD_AMOUNT", "95"))

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global cache for chunks
chunk_cache = {}

# Request models
class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    breakpoint_threshold_type: Optional[str] = Field(default="percentile", pattern="^(percentile|standard_deviation|interquartile)$")
    breakpoint_threshold_amount: Optional[float] = Field(default=95, ge=0, le=100)
    webhook_url: Optional[str] = None

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class BatchChunkRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10)
    breakpoint_threshold_type: Optional[str] = Field(default="percentile", pattern="^(percentile|standard_deviation|interquartile)$")
    breakpoint_threshold_amount: Optional[float] = Field(default=95, ge=0, le=100)

class ChunkResponse(BaseModel):
    chunks: List[str]
    metadata: Dict[str, Any]

# Semantic Chunker implementation using LangChain
class LangChainSemanticChunker:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        
    def create_chunker(self, breakpoint_threshold_type: str = "percentile", breakpoint_threshold_amount: float = 95) -> SemanticChunker:
        """Create a semantic chunker with specified parameters."""
        return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    
    def chunk_text(self, text: str, breakpoint_threshold_type: str = "percentile", breakpoint_threshold_amount: float = 95) -> List[str]:
        """Chunk text using LangChain's SemanticChunker."""
        try:
            chunker = self.create_chunker(breakpoint_threshold_type, breakpoint_threshold_amount)
            
            # Split the text into chunks
            chunks = chunker.split_text(text)
            
            logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def batch_chunk_texts(self, texts: List[str], breakpoint_threshold_type: str = "percentile", breakpoint_threshold_amount: float = 95) -> List[List[str]]:
        """Chunk multiple texts using the same chunker configuration."""
        try:
            chunker = self.create_chunker(breakpoint_threshold_type, breakpoint_threshold_amount)
            
            results = []
            for text in texts:
                chunks = chunker.split_text(text)
                results.append(chunks)
                
            logger.info(f"Processed {len(texts)} texts into chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error batch chunking texts: {e}")
            raise

# Cache helper functions
def get_text_hash(text: str, breakpoint_threshold_type: str = "percentile", breakpoint_threshold_amount: float = 95) -> str:
    """Generate a hash for caching that includes chunking parameters."""
    cache_key = f"{text}_{breakpoint_threshold_type}_{breakpoint_threshold_amount}"
    return hashlib.md5(cache_key.encode()).hexdigest()

@lru_cache(maxsize=CACHE_SIZE)
def get_cached_result(text_hash: str) -> Optional[Dict[str, Any]]:
    return chunk_cache.get(text_hash)

def cache_result(text_hash: str, result: Dict[str, Any]):
    if len(chunk_cache) >= CACHE_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(chunk_cache))
        del chunk_cache[oldest_key]
    chunk_cache[text_hash] = result

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Semantic Chunking Service")
    yield
    # Shutdown
    logger.info("Shutting down Semantic Chunking Service")
    gc.collect()

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Chunking Service",
    description="LLM-based semantic text chunking service for n8n integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware for garbage collection
@app.middleware("http")
async def garbage_collect_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        gc.collect()
    return response

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_size": len(chunk_cache),
        "version": "1.0.0"
    }

# Main chunking endpoint
@app.post("/api/chunk", response_model=ChunkResponse)
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def chunk_text(request: Request, chunk_request: ChunkRequest):
    try:
        # Validate OpenAI API key is configured
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Check cache first
        text_hash = get_text_hash(
            chunk_request.text,
            chunk_request.breakpoint_threshold_type,
            chunk_request.breakpoint_threshold_amount
        )
        cached_result = get_cached_result(text_hash)
        
        if cached_result:
            logger.info(f"Cache hit for text hash: {text_hash}")
            return ChunkResponse(**cached_result)
        
        # Process text using LangChain SemanticChunker
        chunker = LangChainSemanticChunker(embeddings)
        chunks = chunker.chunk_text(
            chunk_request.text,
            chunk_request.breakpoint_threshold_type,
            chunk_request.breakpoint_threshold_amount
        )
        
        # Prepare response
        result = {
            "chunks": chunks,
            "metadata": {
                "total_chunks": len(chunks),
                "original_length": len(chunk_request.text),
                "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "breakpoint_threshold_type": chunk_request.breakpoint_threshold_type,
                "breakpoint_threshold_amount": chunk_request.breakpoint_threshold_amount,
                "processing_time": datetime.utcnow().isoformat()
            }
        }
        
        # Cache result
        cache_result(text_hash, result)
        
        logger.info(f"Processed text into {len(chunks)} chunks using {chunk_request.breakpoint_threshold_type} threshold")
        
        return ChunkResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing chunk request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch chunking endpoint
@app.post("/api/batch-chunk")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE//2}/minute")
async def batch_chunk_text(request: Request, batch_request: BatchChunkRequest):
    try:
        # Validate OpenAI API key is configured
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Process texts using LangChain SemanticChunker
        chunker = LangChainSemanticChunker(embeddings)
        chunks_list = chunker.batch_chunk_texts(
            batch_request.texts,
            batch_request.breakpoint_threshold_type,
            batch_request.breakpoint_threshold_amount
        )
        
        results = []
        for i, chunks in enumerate(chunks_list):
            results.append({
                "index": i,
                "chunks": chunks,
                "metadata": {
                    "total_chunks": len(chunks),
                    "original_length": len(batch_request.texts[i]),
                    "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0
                }
            })
        
        return {
            "results": results,
            "total_texts": len(batch_request.texts),
            "breakpoint_threshold_type": batch_request.breakpoint_threshold_type,
            "breakpoint_threshold_amount": batch_request.breakpoint_threshold_amount,
            "processing_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing batch chunk request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Semantic Chunking Service",
        "endpoints": {
            "health": "/api/health",
            "chunk": "/api/chunk",
            "batch_chunk": "/api/batch-chunk"
        }
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)