# Semantic Chunking Service

A lightweight, production-ready semantic text chunking service built with FastAPI and OpenAI's GPT-4o-mini, designed for deployment on render.com's free tier and seamless integration with n8n workflows.

## Features

- **Agentic Chunking**: Uses GPT-4o-mini to intelligently split text based on semantic meaning
- **n8n Integration**: REST API endpoints optimized for n8n workflow automation
- **Memory Optimized**: Designed for render.com's free tier (256MB RAM, 0.1 CPU)
- **Caching System**: LRU cache for improved performance and reduced API costs
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Batch Processing**: Support for processing multiple texts in a single request
- **Health Monitoring**: Comprehensive health checks and logging

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/leex279/dynamous-semantic-chunking-service.git
   cd dynamous-semantic-chunking-service
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run the service**
   ```bash
   python main.py
   ```

The service will be available at `http://localhost:8000`

### Deploy to Render.com

1. **Fork this repository**
2. **Connect to Render.com**
3. **Add environment variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
4. **Deploy using the included render.yaml**

## API Endpoints

### POST `/api/chunk`
Chunk a single text into semantic segments.

**Request:**
```json
{
  "text": "Your text to chunk here...",
  "max_chunk_size": 1000,
  "api_key": "your_api_key"
}
```

**Response:**
```json
{
  "chunks": ["First semantic chunk...", "Second semantic chunk..."],
  "metadata": {
    "total_chunks": 2,
    "original_length": 500,
    "avg_chunk_size": 250,
    "processing_time": "2024-01-01T12:00:00Z"
  }
}
```

### POST `/api/batch-chunk`
Process multiple texts in a single request.

**Request:**
```json
{
  "texts": ["First text...", "Second text..."],
  "max_chunk_size": 1000,
  "api_key": "your_api_key"
}
```

### GET `/api/health`
Health check endpoint.

## n8n Integration

### Example Workflow

```json
{
  "nodes": [
    {
      "name": "Input Text",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "chunk-text",
        "responseMode": "lastNode",
        "httpMethod": "POST"
      }
    },
    {
      "name": "Semantic Chunking",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "https://your-service.onrender.com/api/chunk",
        "jsonParameters": true,
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
        "functionCode": "return items[0].json.chunks.map(chunk => ({json: {chunk}}))"
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

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute | 60 |
| `MAX_TEXT_LENGTH` | Maximum text length | 50000 |
| `MAX_CHUNK_SIZE` | Maximum chunk size | 1000 |
| `CACHE_SIZE` | Number of cached results | 100 |
| `LOG_LEVEL` | Logging level | INFO |

## Performance

### Benchmarks
- **Startup Time**: 30-60 seconds (render.com spin-up)
- **Small texts** (<1000 chars): <2 seconds
- **Medium texts** (1000-5000 chars): 2-5 seconds
- **Large texts** (>5000 chars): 5-15 seconds

### Resource Usage
- **Memory**: 150-200MB typical usage
- **CPU**: 80-90% during processing
- **Cache Hit Rate**: 30-50% for common content

## Cost Optimization

- **GPT-4o-mini pricing**: ~$0.001-0.003 per 1000-word document
- **Intelligent caching**: Reduces repeated API calls
- **Batch processing**: Efficient for multiple texts
- **Request coalescing**: Optimizes API usage

## Security

- API key validation
- Rate limiting per endpoint
- Input size validation
- Secure environment variable handling

## Development

### Project Structure
```
semantic-chunking-service/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── render.yaml         # Render.com deployment config
├── .env.example        # Environment variables template
├── PLANNING.md         # Implementation plan
└── README.md          # This file
```

### Local Testing
```bash
# Start the service
python main.py

# Test health endpoint
curl http://localhost:8000/api/health

# Test chunking endpoint
curl -X POST http://localhost:8000/api/chunk \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test text here", "api_key": "test"}'
```

## Monitoring

The service includes comprehensive logging and health checks:
- Request/response logging
- Error tracking
- Performance metrics
- Cache statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check the PLANNING.md for implementation details
- Review the logs for troubleshooting