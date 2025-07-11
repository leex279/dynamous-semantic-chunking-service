# Semantic Chunking Service - Implementation Plan

## Project Overview

This document outlines the comprehensive implementation plan for a semantic chunking service designed to:
- Deploy on render.com's free tier (256MB RAM, 0.1 CPU)
- Integrate seamlessly with n8n workflows
- Implement agentic LLM-based chunking for superior semantic coherence
- Optimize for resource constraints while maintaining high quality

## Architecture Design

### System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│    n8n      │────▶│ Semantic Chunking│────▶│  OpenAI API │
│  Workflow   │◀────│     Service      │◀────│   (GPT-4)   │
└─────────────┘     └──────────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ In-Memory    │
                    │    Cache     │
                    └──────────────┘
```

### Core Components

1. **FastAPI Application**: Lightweight async web framework
2. **Agentic Chunker**: LLM-based semantic chunking logic
3. **Cache Layer**: LRU cache for repeated content
4. **Queue Manager**: Handle requests during spin-up
5. **Rate Limiter**: Prevent API abuse

## Technical Implementation

### 1. Agentic Chunking Algorithm

Based on the provided approach, our implementation will:

```python
class AgenticChunker:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.chunks = []
    
    def extract_propositions(self, text: str) -> List[str]:
        # Extract stand-alone statements using LLM
        prompt = f"""Extract stand-alone propositions from this text.
        Each proposition should be self-contained.
        
        Text: {text}
        
        Return a list of propositions, one per line."""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().split('\n')
    
    def should_add_to_chunk(self, proposition: str, chunk: str) -> bool:
        # Determine semantic relationship
        prompt = f"""Determine if this proposition belongs with this chunk.
        
        Chunk: {chunk}
        Proposition: {proposition}
        
        Return 'yes' if they are semantically related, 'no' otherwise."""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower() == 'yes'
```

### 2. API Endpoints

```python
# Main chunking endpoint
@app.post("/api/chunk")
async def chunk_text(request: ChunkRequest):
    """
    Process text into semantic chunks
    Input: {
        "text": str,
        "max_chunk_size": int (optional, default: 1000),
        "api_key": str,
        "webhook_url": str (optional, for async processing)
    }
    """

# Batch processing endpoint
@app.post("/api/batch-chunk")
async def batch_chunk(request: BatchChunkRequest):
    """
    Process multiple texts in a single request
    """

# Health check
@app.get("/api/health")
async def health_check():
    """
    Monitor service health and prevent spin-downs
    """
```

### 3. Memory Optimization Strategies

Given the 256MB RAM constraint:

1. **Stream Processing**:
   - Process text in chunks rather than loading entire documents
   - Use generators for large text processing

2. **Aggressive Garbage Collection**:
   ```python
   import gc
   
   @app.middleware("http")
   async def garbage_collect(request: Request, call_next):
       response = await call_next(request)
       gc.collect()
       return response
   ```

3. **Limited Cache Size**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)  # Limit cache entries
   def cached_chunk_result(text_hash: str):
       # Cache results for repeated content
   ```

### 4. Performance Optimizations

1. **Request Coalescing**:
   - Batch similar requests together
   - Reduce API calls to OpenAI

2. **Async Processing**:
   - Use FastAPI's async capabilities
   - Non-blocking I/O for API calls

3. **Compression**:
   ```python
   from gzip import compress, decompress
   
   # Compress large responses
   if len(response_data) > 1000:
       compressed = compress(json.dumps(response_data).encode())
   ```

## Deployment Configuration

### render.yaml

```yaml
services:
  - type: web
    name: semantic-chunking-service
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: RATE_LIMIT_PER_MINUTE
        value: 60
      - key: MAX_TEXT_LENGTH
        value: 50000
    healthCheckPath: /api/health
    numInstances: 1
    plan: free
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only essential dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Minimize image size
RUN find . -type d -name __pycache__ -exec rm -r {} +
RUN find . -type f -name "*.pyc" -delete

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## n8n Integration

### Workflow Example

```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "chunk-text",
        "responseMode": "lastNode",
        "httpMethod": "POST"
      }
    },
    {
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "https://your-service.onrender.com/api/chunk",
        "jsonParameters": true,
        "options": {},
        "bodyParametersJson": {
          "text": "={{$json.text}}",
          "api_key": "={{$credentials.apiKey}}"
        }
      }
    },
    {
      "name": "Process Chunks",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "// Process chunks for embedding\nreturn items[0].json.chunks.map(chunk => ({json: {chunk}}))"
      }
    },
    {
      "name": "Embeddings OpenAI",
      "type": "@n8n/n8n-nodes-langchain.embeddingsOpenAi",
      "parameters": {
        "model": "text-embedding-3-small"
      }
    }
  ]
}
```

## Performance Benchmarks

### Expected Performance

- **Startup Time**: 30-60 seconds (render.com spin-up)
- **Chunking Speed**: 
  - Small texts (<1000 chars): <2 seconds
  - Medium texts (1000-5000 chars): 2-5 seconds
  - Large texts (>5000 chars): 5-15 seconds
- **Concurrent Requests**: 5-10 (limited by CPU)
- **Cache Hit Rate**: Target 30-50% for common content

### Resource Usage

- **Memory**: 150-200MB typical usage
- **CPU**: 80-90% during processing
- **Network**: Minimal (API calls only)

## Monitoring & Maintenance

### Health Checks

```python
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "cache_size": len(cache),
        "memory_usage": get_memory_usage(),
        "uptime": get_uptime()
    }
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log key metrics
logger.info(f"Chunk request: size={len(text)}, chunks={len(chunks)}")
```

## Security Considerations

1. **API Key Validation**:
   ```python
   def validate_api_key(api_key: str):
       # Implement secure key validation
       return api_key in valid_keys
   ```

2. **Input Sanitization**:
   - Limit text size to prevent DoS
   - Validate input format
   - Escape special characters

3. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/api/chunk")
   @limiter.limit("60/minute")
   async def chunk_text(request: ChunkRequest):
       # Process request
   ```

## Cost Analysis

### OpenAI API Costs (GPT-4o-mini)
- Input: $0.00015 per 1K tokens
- Output: $0.0006 per 1K tokens
- Estimated cost per 1000-word document: $0.001-0.003

### Optimization Strategies
1. Use caching aggressively
2. Batch similar propositions
3. Implement request coalescing
4. Monitor usage patterns

## Implementation Timeline

### Phase 1: Core Development (Week 1)
- [ ] Set up FastAPI project structure
- [ ] Implement basic chunking logic
- [ ] Add OpenAI integration
- [ ] Create API endpoints

### Phase 2: Optimization (Week 2)
- [ ] Implement caching layer
- [ ] Add rate limiting
- [ ] Optimize memory usage
- [ ] Add batch processing

### Phase 3: Deployment (Week 3)
- [ ] Create Dockerfile
- [ ] Configure render.yaml
- [ ] Deploy to render.com
- [ ] Test with n8n

### Phase 4: Monitoring & Enhancement (Week 4)
- [ ] Add comprehensive logging
- [ ] Implement monitoring
- [ ] Performance tuning
- [ ] Documentation

## Future Enhancements

1. **Alternative LLM Support**:
   - Add support for Claude, Gemini
   - Implement fallback mechanisms

2. **Advanced Chunking**:
   - Hierarchical chunking
   - Domain-specific chunking rules
   - Multi-language support

3. **Performance**:
   - Redis caching (if budget allows)
   - WebSocket support for real-time processing
   - GPU acceleration for embeddings

4. **Integration**:
   - Direct database connectors
   - S3/cloud storage support
   - Webhook callbacks

## Conclusion

This implementation plan provides a robust foundation for building a semantic chunking service that:
- Works within render.com's free tier constraints
- Integrates seamlessly with n8n workflows
- Provides high-quality semantic chunking using LLMs
- Scales efficiently within resource limits

The modular design allows for future enhancements while maintaining core functionality on limited resources.