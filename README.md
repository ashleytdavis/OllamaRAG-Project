# Ollama RAG Ingest and Search

## Set Up Instructions

- Ollama app set up ([Ollama.com](Ollama.com))
- Install the necessary requirements by running the following:
```
pip install requirements.txt
```
- Redis Stack running on port 6379.  
If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.
Run the following command to create and run a redis container on docker:
```
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```
- ChromaDB instance running on port 8000
Run the following command to create and run a chromadb container on docker:
```
docker run -d --rm --name chromadb -p 8000:8000 -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE chromadb/chroma:0.6.3
```
- Our implementation tests multiple LLM models offerred by ollama - they
can be installed by running the following:
```
ollama run llama3.2
ollama run mistral
ollama run deepseek-r1
```

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 