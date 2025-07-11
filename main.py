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
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
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

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global cache for chunks
chunk_cache = {}

# Request models
class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    max_chunk_size: Optional[int] = Field(default=MAX_CHUNK_SIZE, ge=100, le=5000)
    api_key: str = Field(..., min_length=1)
    webhook_url: Optional[str] = None

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class BatchChunkRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10)
    max_chunk_size: Optional[int] = Field(default=MAX_CHUNK_SIZE, ge=100, le=5000)
    api_key: str = Field(..., min_length=1)

class ChunkResponse(BaseModel):
    chunks: List[str]
    metadata: Dict[str, Any]

# Agentic Chunker implementation
class AgenticChunker:
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
        self.chunks = []
        
    async def extract_propositions(self, text: str) -> List[str]:
        try:
            prompt = f"""Extract stand-alone propositions from this text.
            Each proposition should be self-contained and meaningful.
            
            Text: {text}
            
            Return a list of propositions, one per line. Be concise."""
            
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            propositions = [p.strip() for p in content.split('\n') if p.strip()]
            
            logger.info(f"Extracted {len(propositions)} propositions from text of length {len(text)}")
            return propositions
            
        except Exception as e:
            logger.error(f"Error extracting propositions: {e}")
            raise
    
    async def should_add_to_chunk(self, proposition: str, chunk: str) -> bool:
        try:
            prompt = f"""Determine if this proposition belongs with this chunk based on semantic similarity.
            
            Chunk: {chunk}
            Proposition: {proposition}
            
            Return only 'yes' or 'no'."""
            
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            return result == 'yes'
            
        except Exception as e:
            logger.error(f"Error checking chunk similarity: {e}")
            return False
    
    async def chunk_text(self, text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
        self.chunks = []
        
        # Split text into manageable sections if it's too large
        sections = self._split_into_sections(text, max_length=3000)
        
        for section in sections:
            propositions = await self.extract_propositions(section)
            
            for prop in propositions:
                if not prop:
                    continue
                    
                added = False
                
                # Check against existing chunks
                for i, chunk in enumerate(self.chunks):
                    # Skip if chunk is already too large
                    if len(chunk) + len(prop) + 1 > max_chunk_size:
                        continue
                        
                    if await self.should_add_to_chunk(prop, chunk):
                        self.chunks[i] = f"{chunk} {prop}"
                        added = True
                        break
                
                # Create new chunk if not added
                if not added:
                    self.chunks.append(prop)
        
        # Merge small chunks if possible
        self.chunks = await self._merge_small_chunks(self.chunks, max_chunk_size)
        
        return self.chunks
    
    def _split_into_sections(self, text: str, max_length: int = 3000) -> List[str]:
        if len(text) <= max_length:
            return [text]
            
        sections = []
        sentences = text.split('. ')
        current_section = ""
        
        for sentence in sentences:
            if len(current_section) + len(sentence) + 2 > max_length:
                sections.append(current_section.strip())
                current_section = sentence
            else:
                current_section += ". " + sentence if current_section else sentence
                
        if current_section:
            sections.append(current_section.strip())
            
        return sections
    
    async def _merge_small_chunks(self, chunks: List[str], max_size: int) -> List[str]:
        if len(chunks) <= 1:
            return chunks
            
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Try to merge with next chunks if current is small
            if len(current_chunk) < max_size // 2 and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                if len(current_chunk) + len(next_chunk) + 1 <= max_size:
                    # Check if they should be merged
                    if await self.should_add_to_chunk(next_chunk, current_chunk):
                        current_chunk = f"{current_chunk} {next_chunk}"
                        i += 1
            
            merged.append(current_chunk)
            i += 1
            
        return merged

# Cache helper functions
def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

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
        # Validate API key (simple check for demo)
        if not chunk_request.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check cache first
        text_hash = get_text_hash(chunk_request.text)
        cached_result = get_cached_result(text_hash)
        
        if cached_result:
            logger.info(f"Cache hit for text hash: {text_hash}")
            return ChunkResponse(**cached_result)
        
        # Process text
        chunker = AgenticChunker(openai_client)
        chunks = await chunker.chunk_text(
            chunk_request.text,
            chunk_request.max_chunk_size
        )
        
        # Prepare response
        result = {
            "chunks": chunks,
            "metadata": {
                "total_chunks": len(chunks),
                "original_length": len(chunk_request.text),
                "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "processing_time": datetime.utcnow().isoformat()
            }
        }
        
        # Cache result
        cache_result(text_hash, result)
        
        logger.info(f"Processed text into {len(chunks)} chunks")
        
        return ChunkResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing chunk request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch chunking endpoint
@app.post("/api/batch-chunk")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE//2}/minute")
async def batch_chunk_text(request: Request, batch_request: BatchChunkRequest):
    try:
        # Validate API key
        if not batch_request.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        results = []
        chunker = AgenticChunker(openai_client)
        
        # Process texts concurrently
        tasks = []
        for text in batch_request.texts:
            task = chunker.chunk_text(text, batch_request.max_chunk_size)
            tasks.append(task)
        
        chunks_list = await asyncio.gather(*tasks)
        
        for i, chunks in enumerate(chunks_list):
            results.append({
                "index": i,
                "chunks": chunks,
                "metadata": {
                    "total_chunks": len(chunks),
                    "original_length": len(batch_request.texts[i])
                }
            })
        
        return {
            "results": results,
            "total_texts": len(batch_request.texts),
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