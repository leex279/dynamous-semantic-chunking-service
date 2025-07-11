# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Local Development Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with required API keys
```

### Running the Service
```bash
# Start the FastAPI service
python main.py

# The service runs on port 8000 by default
# Use PORT environment variable to change: export PORT=8001
```

### Testing
```bash
# Test API endpoints manually
python test_api.py

# Test semantic chunker functionality
python test_semantic_chunker.py

# Health check
curl http://localhost:8000/api/health

# Test chunking with authentication
curl -X POST http://localhost:8000/api/chunk \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test text here"}'
```

### Deployment
```bash
# Deploy to render.com using the included render.yaml
# Set required environment variables:
# - OPENAI_API_KEY
# - API_KEYS (format: key1:user1,key2:user2)
# - ALLOWED_ORIGINS (optional)
```

## Architecture Overview

### Core System Design
This is a FastAPI-based semantic text chunking service that uses LangChain's SemanticChunker with OpenAI embeddings. The architecture follows a three-layer pattern:

1. **API Layer** (`main.py`): FastAPI application with authentication, rate limiting, and CORS
2. **Processing Layer** (`LangChainSemanticChunker` class): Wrapper around LangChain's SemanticChunker
3. **External Services**: OpenAI embeddings API for semantic similarity analysis

### Security Architecture
The service implements enterprise-grade security with:
- **API Key Authentication**: Bearer token authentication using environment-configured keys
- **Per-API-Key Rate Limiting**: Individual hourly limits tracked in memory (`api_key_usage` dict)
- **CORS Security**: Configurable allowed origins and restricted headers
- **Audit Logging**: All requests logged with user identification and usage metrics

### Memory Management
Designed for render.com's free tier (256MB RAM):
- **LRU Cache**: `chunk_cache` dict with configurable size limit
- **Garbage Collection**: Aggressive GC after each API request
- **Request Coalescing**: Batch processing support to minimize memory usage

### Key Components

#### Authentication Flow
1. `parse_api_keys()` parses environment variable format `key1:user1,key2:user2`
2. `authenticate_api_key()` validates Bearer tokens against parsed keys
3. `check_api_key_rate_limit()` enforces per-key hourly limits
4. `authenticate_and_rate_limit()` combines auth + rate limiting for endpoints

#### Chunking Pipeline
1. Input validation via Pydantic models (`ChunkRequest`, `BatchChunkRequest`)
2. Cache lookup using `get_text_hash()` with chunking parameters
3. `LangChainSemanticChunker.chunk_text()` processes text with configurable thresholds
4. Results cached and returned with metadata

#### Configuration Management
All configuration via environment variables with defaults:
- `OPENAI_API_KEY`: Required for embeddings
- `API_KEYS`: Security keys (required for auth)
- `RATE_LIMIT_PER_KEY_HOUR`: Per-key rate limit (default: 100)
- `BREAKPOINT_THRESHOLD_TYPE`: Chunking algorithm (percentile/standard_deviation/interquartile)
- `ALLOWED_ORIGINS`: CORS origins (default: "*")

### n8n Integration Points
The service is specifically designed for n8n workflow integration:
- REST API with JSON responses compatible with n8n HTTP Request nodes
- Bearer token authentication using n8n credential system
- Webhook support for async processing (optional `webhook_url` parameter)
- Batch processing endpoint for handling multiple texts in n8n workflows

### Error Handling Strategy
- HTTP 401 for authentication failures
- HTTP 429 for rate limit violations  
- HTTP 500 for OpenAI API or processing errors
- Comprehensive logging for debugging and monitoring
- Graceful degradation when OpenAI API is unavailable

### Performance Considerations
- Text processing optimized for render.com's 0.1 CPU allocation
- Caching strategy reduces OpenAI API calls and costs
- Memory usage monitored and garbage collected aggressively
- Async FastAPI for concurrent request handling within memory constraints