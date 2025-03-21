# Ollama RAG Ingest and Search

## Set Up Instructions

- Ollama app set up ([Ollama.com](Ollama.com))
- Install the necessary requirements by running the following:
```
pip install requirements.txt
```
- Redis Stack running (Docker container is fine) on port 6379.  
If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.
Run the following command to create and run a redis container on docker:
```
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 
- `src/trees.py` - a simple q/a command-line based system that is designated to answering tree-based questions
