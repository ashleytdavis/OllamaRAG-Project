import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import re
from collections import deque

# --- Tree Structures Implementation ---

# Binary Search Tree Implementation
class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def insert(self, key):
        if key < self.key:
            if self.left is None:
                self.left = BSTNode(key)
            else:
                self.left.insert(key)
        else:
            if self.right is None:
                self.right = BSTNode(key)
            else:
                self.right.insert(key)


def build_bst(numbers):
    if not numbers:
        return None
    root = BSTNode(numbers[0])
    for num in numbers[1:]:
        root.insert(num)
    return root

# AVL Tree Implementation
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


def get_height(node):
    if not node:
        return 0
    return node.height


def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)


def right_rotate(z):
    y = z.left
    T3 = y.right
    y.right = z
    z.left = T3
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y


def left_rotate(z):
    y = z.right
    T2 = y.left
    y.left = z
    z.right = T2
    z.height = 1 + max(get_height(z.left), get_height(z.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y


def insert_avl(node, key):
    if not node:
        return AVLNode(key)
    if key < node.key:
        node.left = insert_avl(node.left, key)
    else:
        node.right = insert_avl(node.right, key)

    node.height = 1 + max(get_height(node.left), get_height(node.right))
    balance = get_balance(node)

    # Left Left
    if balance > 1 and key < node.left.key:
        return right_rotate(node)
    # Right Right
    if balance < -1 and key > node.right.key:
        return left_rotate(node)
    # Left Right
    if balance > 1 and key > node.left.key:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    # Right Left
    if balance < -1 and key < node.right.key:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node


def build_avl_tree(numbers):
    root = None
    for num in numbers:
        root = insert_avl(root, num)
    return root


def tree_to_string(node):
    """Return a string representing the level order traversal of the tree."""
    if node is None:
        return ""
    result = ""
    q = deque([node])
    while q:
        level_size = len(q)
        level_nodes = []
        for _ in range(level_size):
            current = q.popleft()
            level_nodes.append(str(current.key))
            if current.left:
                q.append(current.left)
            if current.right:
                q.append(current.right)
        result += " ".join(level_nodes) + "\n"
    return result

# --- End of Tree Structures Implementation ---

# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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
                f"---> File: {result['file']}, Page: {result['page']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, model):
    global conversation_memory
    #now takes desired model as input

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    conversation_history = "\n".join([
        f"Q: {pair.get('query', '')}\nA: {pair.get('response', '')}"
        for pair in conversation_memory
    ])

    # Construct prompt with context
    prompt = f"""You are an expert programming assistant with access to reference material.
    When answering, follow these steps:

    1. Use only the provided context to answer the question.
    2. If answering a multiple-choice question, select the best answer and justify it with keywords or definitions from the context.
    3. If the answer requires explanation, provide it clearly and concisely, focusing on correctness over creativity.
    4. If the context does not contain enough information to answer confidently, respond with 'I don't know'.
    5. Be precise, factual, and avoid assumptions.
    
    If the context is not relevant to the query, say 'I don't know'.

    Context:
    {context_str}

    Previous Conversation:
    {conversation_history}

    Query: {query}

    Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        #model="mistral:latest", messages=[{"role": "user", "content": prompt}]
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    global conversation_memory
    if 'conversation_memory' not in globals():
        conversation_memory = []
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:

        # Check if the query is asking to build an AVL tree or a binary search tree
        query = input("\nEnter your search query (or type 'reset' to clear memory; 'exit' to exit interface): ")


        if query.lower() == 'exit':
            return

        if query.lower() == 'reset':
            conversation_memory = []
            print('Conersation Memory Cleared')
            continue

        # Check if the query is asking to build an AVL tree or a binary search tree
        if ('avl tree' in query.lower() or 'binary search tree' in query.lower() or 'bst' in query.lower()) and 'build' in query.lower():
            # Look for the first occurrence of '[' and ']'
            start_idx = query.find('[')
            end_idx = query.find(']', start_idx)
            numbers = []
            if start_idx != -1 and end_idx != -1:
                # Extract text between '[' and ']'
                numbers_text = query[start_idx+1:end_idx]
                try:
                    numbers = [int(x.strip()) for x in numbers_text.split(",") if x.strip()]
                except Exception as e:
                    print(f"Error parsing numbers from query: {e}")
                    numbers = []
                if numbers:
                    # Confirm with the user
                    confirmation = input(f"You provided numbers: {numbers}. Do you want to build the tree? (yes/no): ")
                    if confirmation.strip().lower().startswith('y'):
                        if 'avl tree' in query.lower():
                            tree_root = build_avl_tree(numbers)
                            tree_str = tree_to_string(tree_root)
                            print("\n--- AVL Tree ---")
                            print(tree_str)
                        else:
                            tree_root = build_bst(numbers)
                            tree_str = tree_to_string(tree_root)
                            print("\n--- Binary Search Tree ---")
                            print(tree_str)
                        continue
                    else:
                        # Proceed with regular search if user declines
                        pass
                else:
                    # No valid numbers found in the query
                    response = input("No valid numbers found in your query. Do you want to build a tree? (yes/no): ")
                    if response.strip().lower().startswith('y'):
                        numbers_input = input("Please provide a list of numbers separated by commas: ")
                        try:
                            numbers = [int(x.strip()) for x in numbers_input.split(",") if x.strip()]
                        except Exception as e:
                            print(f"Invalid input. Proceeding with regular search. Error: {e}")
                            numbers = []
                        if numbers:
                            if 'avl tree' in query.lower():
                                tree_root = build_avl_tree(numbers)
                                tree_str = tree_to_string(tree_root)
                                print("\n--- AVL Tree ---")
                                print(tree_str)
                            else:
                                tree_root = build_bst(numbers)
                                tree_str = tree_to_string(tree_root)
                                print("\n--- Binary Search Tree ---")
                                print(tree_str)
                            continue
                        else:
                            # No numbers provided, proceed with regular search
                            pass
                    else:
                        # User declined to build a tree, proceed with regular search
                        pass
            else:
                # No brackets found in the query
                response = input("No numbers provided in query. Do you want to build a tree? (yes/no): ")
                if response.strip().lower().startswith('y'):
                    numbers_input = input("Please provide a list of numbers separated by commas: ")
                    try:
                        numbers = [int(x.strip()) for x in numbers_input.split(",") if x.strip()]
                    except Exception as e:
                        print(f"Invalid input. Proceeding with regular search. Error: {e}")
                        numbers = []
                    if numbers:
                        if 'avl tree' in query.lower():
                            tree_root = build_avl_tree(numbers)
                            tree_str = tree_to_string(tree_root)
                            print("\n--- AVL Tree ---")
                            print(tree_str)
                        else:
                            tree_root = build_bst(numbers)
                            tree_str = tree_to_string(tree_root)
                            print("\n--- Binary Search Tree ---")
                            print(tree_str)
                        continue
                    else:
                        pass
                else:
                    pass

        #select model
        model = 'mistral:latest'

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results, model)
        safe_model = re.sub(r'[\/:*?"<>|&]', '_', model)
        print("\n--- Response ---")
        print(response)

        # Save the conversation pair to memory
        conversation_memory.append({
            'query': query,
            'response': response
        })

        with open(f"{safe_model}.txt", "a") as file:
            file.write(query+ '\n')
            file.write(response)



# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )


if __name__ == "__main__":
    interactive_search()
