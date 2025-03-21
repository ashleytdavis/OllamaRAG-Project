import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField

'''
    Extremely similar to that of search.py, but this implementation
    is designed for tree-based questions for our midterm.
'''


# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""
        You are an expert programming assistant with access to reference material on tree-based data structures (AVL trees, B-trees, B+ trees, binary search trees, and related concepts).

        When answering, follow these steps:
        1. Use only the provided context to answer the question.
        2. Focus on correctness, precision, and definitions, especially:
        - tree properties (height, balance factors, order)
        - insertion, deletion, and balancing procedures
        - advantages and typical use cases
        3. If answering a multiple-choice question, select the best answer and justify it using keywords or definitions from the context (e.g., 'balance factor in AVL', 'order of B+ tree', 'leaf node property').
        4. If an explanation is required, provide a clear, step-by-step description, referencing procedures or properties from the context.
        5. If the question asks for a tree diagram, draw it using clean ASCII representation. Keep the diagram simple, properly indented, and easy to follow.
        6. If the context does not contain enough information to answer confidently, respond with 'I don't know'.
        7. Be factual, precise, and avoid assumptions. 
        8. If the question asks for comparisons, list differences in bullet points using definitions or properties from the context.

        If the context is not relevant to tree-based structures, say 'I don't know'.

        Context:
        {context_str}

        Query: {query}

        Answer:
        """


    # Generate response using Ollama
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Tree Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your tree-based query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
