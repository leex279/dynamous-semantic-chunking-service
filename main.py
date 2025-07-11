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

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
RATE_LIMIT_PER_KEY_HOUR = int(os.getenv("RATE_LIMIT_PER_KEY_HOUR", "100"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))
BREAKPOINT_THRESHOLD_TYPE = os.getenv("BREAKPOINT_THRESHOLD_TYPE", "percentile")
BREAKPOINT_THRESHOLD_AMOUNT = float(os.getenv("BREAKPOINT_THRESHOLD_AMOUNT", "95"))
API_KEYS = os.getenv("API_KEYS", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global cache for chunks
chunk_cache = {}

# Parse API keys from environment
def parse_api_keys(api_keys_str: str) -> Dict[str, str]:
    """Parse API keys from environment variable format: key1:user1,key2:user2"""
    if not api_keys_str:
        return {}
    
    api_keys = {}
    for key_pair in api_keys_str.split(','):
        if ':' in key_pair:
            key, identifier = key_pair.strip().split(':', 1)
            api_keys[key.strip()] = identifier.strip()
    return api_keys

valid_api_keys = parse_api_keys(API_KEYS)

# Parse allowed origins
allowed_origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(',') if origin.strip()]
if ALLOWED_ORIGINS == "*":
    allowed_origins = ["*"]

# Initialize HTTP Bearer security
security = HTTPBearer()

# Rate limiting tracking per API key
api_key_usage = {}

# Authentication dependency
async def authenticate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate API key and return user identifier"""
    api_key = credentials.credentials
    
    if api_key not in valid_api_keys:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = valid_api_keys[api_key]
    logger.info(f"Authenticated user: {user_id}")
    return user_id

# Per-API-key rate limiting
from time import time

def check_api_key_rate_limit(api_key: str, user_id: str) -> bool:
    """Check if API key is within rate limit"""
    current_time = time()
    hour_start = int(current_time // 3600) * 3600
    
    if api_key not in api_key_usage:
        api_key_usage[api_key] = {}
    
    if hour_start not in api_key_usage[api_key]:
        # Clean old hours and start new one
        api_key_usage[api_key] = {hour_start: 0}
    
    current_hour_requests = api_key_usage[api_key].get(hour_start, 0)
    
    if current_hour_requests >= RATE_LIMIT_PER_KEY_HOUR:
        logger.warning(f"Rate limit exceeded for user {user_id}: {current_hour_requests} requests this hour")
        return False
    
    # Increment counter
    api_key_usage[api_key][hour_start] = current_hour_requests + 1
    return True

# Combined authentication and rate limiting dependency
async def authenticate_and_rate_limit(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Authenticate API key and check rate limits"""
    api_key = credentials.credentials
    user_id = await authenticate_api_key(credentials)
    
    if not check_api_key_rate_limit(api_key, user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_KEY_HOUR} requests per hour.",
        )
    
    return user_id

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
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
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
async def chunk_text(
    chunk_request: ChunkRequest, 
    user_id: str = Depends(authenticate_and_rate_limit)
):
    try:
        # Audit logging
        logger.info(f"Chunk request from user {user_id}: text_length={len(chunk_request.text)}, threshold_type={chunk_request.breakpoint_threshold_type}")
        
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
async def batch_chunk_text(
    batch_request: BatchChunkRequest,
    user_id: str = Depends(authenticate_and_rate_limit)
):
    try:
        # Audit logging
        total_length = sum(len(text) for text in batch_request.texts)
        logger.info(f"Batch chunk request from user {user_id}: texts_count={len(batch_request.texts)}, total_length={total_length}")
        
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