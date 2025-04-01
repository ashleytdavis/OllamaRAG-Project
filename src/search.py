import redis
import numpy as np
import ollama
import chromadb
import re
from chromadb.config import Settings
from qdrant_client import QdrantClient
from redis.commands.search.query import Query


class UnifiedSearch:
    def __init__(self, db_type: str, embedding_model: str):
        """
        Initializes the UnifiedSearch with a given database type and embedding model.

        Args:
            db_type (str): The database type ("redis", "chroma", or "qdrant").
            embedding_model (str): The embedding model to use (e.g., "nomic-embed-text").
        """
        self.db_type = db_type.lower()
        self.embedding_model = embedding_model

        if self.db_type == "redis":
            # Initialize Redis client
            self.redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
            self.index_name = "embedding_index"
            self.DOC_PREFIX = "doc:"

        elif self.db_type == "chroma":
            # Initialize Chroma client and collection
            self.chroma_client = chromadb.HttpClient(
                settings=Settings(allow_reset=True),
                host="localhost",
                port=8000
            )
            self.collection_name = "pdf_embeddings"
            self.collection = self.chroma_client.get_collection(self.collection_name)

        elif self.db_type == "qdrant":
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
            self.collection_name = "pdf_embeddings"
        else:
            raise ValueError("Unsupported database type provided. Use 'redis', 'chroma', or 'qdrant'.")

    def get_embedding(self, text: str) -> list:
        """
        Gets an embedding for the given text using the specified embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def search(self, query: str, top_k: int = 3):
        embedding = self.get_embedding(query)

        if self.db_type == "redis":
            # Convert the embedding to bytes for Redis
            query_vector = np.array(embedding, dtype=np.float32).tobytes()
            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "file", "page", "chunk", "vector_distance")
                .dialect(2)
            )
            results = self.redis_client.ft(self.index_name).search(q, query_params={"vec": query_vector})
            top_results = [
                {
                    "file": result.file,
                    "page": result.page,
                    "chunk": result.chunk,
                    "similarity": result.vector_distance,
                } for result in results.docs
            ][:top_k]
            return top_results

        elif self.db_type == "chroma":
            # Query Chroma with the embedding and original query text.
            results = self.collection.query(
                query_embeddings=[embedding],
                query_texts=[query],
                n_results=top_k,
            )
            top_results = []
            # Iterate over each list of documents returned (one per query)
            for doc_list in results.get("documents", []):
                for doc in doc_list:
                    parts = doc.split("_page_")
                    if len(parts) == 2:
                        file = parts[0]
                        remainder = parts[1]
                        parts2 = remainder.split("_chunk_")
                        if len(parts2) == 2:
                            page, chunk = parts2
                            top_results.append({
                                "file": file,
                                "page": page,
                                "chunk": chunk,
                                "similarity": None,
                            })
            return top_results

        elif self.db_type == "qdrant":
            # Query Qdrant using the vector search API
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=top_k,
            )
            top_results = []
            for res in results:
                top_results.append({
                    "file": res.payload.get("file", "unknown"),
                    "page": res.payload.get("page", "unknown"),
                    "chunk": res.payload.get("chunk", "unknown"),
                    "similarity": res.score,
                })
            return top_results

    def generate_rag_response(self, query: str, context_results, model: str):
        """
        Generates a RAG (Retrieval-Augmented Generation) response using Ollama.

        Args:
            query (str): The original query.
            context_results (list): The context from search results.
            model (str): The model to use for generating the response.

        Returns:
            str: The generated response.
        """
        context_str = "\n".join(
            [
                f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
                f"with similarity {float(result.get('similarity') or 0):.2f}"
                for result in context_results
            ]
        )
        prompt = f"""You are an expert programming assistant with access to reference material.
        
1. Use only the provided context to answer the question.
2. Answer only with ‘true’ or ‘false’. Do not explain. Do not say anything else.
3. If the context is not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        final_response = response["message"]["content"]
        if "deepseek" in model.lower():
            # Remove any <think>...</think> sections
            final_response = re.sub(r"<think>.*?</think>", "", final_response, flags=re.DOTALL)
            parts = final_response.split("**Answer:**")
            if len(parts) > 1:
                final_response = parts[1].strip()
            else:
                final_response = final_response.strip()
        return final_response

def GenerateResponse(embedding_model,db_type,rag_model, query):
    search_engine = UnifiedSearch(db_type=db_type, embedding_model=embedding_model)

    context_results = search_engine.search(query, top_k=3)

    response = search_engine.generate_rag_response(query, context_results, rag_model)

    return response

def InteractiveSearch(embedding_model,db_type,rag_model):

    while True:
        search_engine = UnifiedSearch(db_type=db_type, embedding_model=embedding_model)
        query = input("Enter your search query: ")

        if query.lower() == 'exit':
            return

        # Perform search
        context_results = search_engine.search(query, top_k=3)

        response = search_engine.generate_rag_response(query, context_results, rag_model)
        print("\n--- RAG Response ---")
        print(response)

# Example usage:
if __name__ == "__main__":


    embedding_model = "all-minilm"
    db_type = "chroma"  # redis or "chroma" or "qdrant"
    rag_model = "mistral"


   # InteractiveSearch(embedding_model, db_type, rag_model)
    print(GenerateResponse(embedding_model,db_type,rag_model, "Which of the following statements best describes the key property of an AVL tree? A) An AVL tree is a binary search tree that allows unlimited differences in the heights of its left and right subtrees. B) An AVL tree is a self-balancing binary search tree that maintains the property that the heights of the left and right subtrees of any node differ by no more than 1. C) An AVL tree is a type of binary search tree that only requires balance at the root node. D) An AVL tree is a binary tree that does not use rotations to achieve balance."))
    #print(GenerateResponse(embedding_model,db_type,rag_model, "Who is mishas best friend? A) John B) Joe C) Hannah D) Mia"))
