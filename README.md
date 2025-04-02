<div align="center">
    <img src="https://images.cooltext.com/5727475.png" alt="Header" />
</div>

### Ashley Davis, Misha Ankudovych, Kevin Martone, and Karen Phung

Table of Contents:
- <a href="#About">About</a>
- <a href="#Set-Up-Instructions">Set Up Instructions</a>
- <a href="#Source-Code-Breakdown">Source Code Breakdown</a>
- <a href="#Command-Line-Usage">Command Line Usage</a>
    - <a href="#Indexing-Files">Indexing Files</a>
    - <a href="#Searching-Terms">Searching Terms</a>
    - <a href="#Need-Help">Need Help?</a>
- <a href="#Findings">Results</a>
- <a href="#Project-Authors">Project Authors</a>


## About
This project demonstrates the integration of advanced text processing, embedding generation, and vector database storage using ChromaDB and Redis. It processes PDF files by extracting text, splitting it into manageable chunks, and generating embeddings using various LLM models provided by Ollama. The embeddings are then stored in either Redis or ChromaDB for efficient querying and retrieval. This implementation showcases the power of combining modern vector databases with natural language processing techniques to enable fast and accurate information retrieval.

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
- ChromaDB instance running on port 8000.
Run the following command to create and run a chromadb container on docker:
```
docker run -d --rm --name chromadb -p 8000:8000 -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE chromadb/chroma:0.6.3
```
- Qdrant instance running on port 6333.
Run the following command to create and run a qdrant container on docker:
```
docker run -d --name qdrant_container -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
- Our implementation tests multiple LLM models offerred by ollama - they
can be installed by running the following:
```
ollama run llama3.2
ollama run mistral
ollama run deepseek-r1
```
- Our implementation also tests multiple embedding models offered by ollama - they
can be installed by running the following:
```
ollama pull all-minilm
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

## Command Line Usage
(insert user instructions here)

## Source Code Breakdown
- `src/redis_ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/chroma_ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in ChromaDB
- `src/qdrant_ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Qdrant
- `src/text_process.py` - abstracted pdf and text processing functionality

## Findings
A detailed report about our findings can be found <a href="https://docs.google.com/presentation/d/18PCLwp3W9CTwPKSxhj5DruVK9qEPUlvKcswCLIGUPe0/edit?usp=sharing">here</a>

## Project Authors
Ashley Davis | davis.ash@northeastern.edu | [Github](https://github.com/ashleytdavis)

Misha Ankudovych | ankudovych.m@northeastern.edu | [Github](https://github.com/ankudovychm)

Kevin Martone | martone.k@northeastern.edu | [Github](https://github.com/kevinmartone)

Karen Phung | phung.ka@northeastern.edu | [Github](https://github.com/karenphung01)
