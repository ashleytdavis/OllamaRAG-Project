import os
import uuid
import numpy as np
import redis
import chromadb
import re
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from redis.commands.search.query import Query
from src.text_process import TextProcess

COLLECTION_NAME = "pdf_embeddings"


class UnifiedIngest:
    def __init__(self, db_type: str, embedding_model: str, preprocess: bool = False):
        """
        Initializes the ingestion class for a specified DB type, embedding model, and text preprocessing flag.

        Args:
            db_type (str): The type of database ("chroma", "qdrant", or "redis").
            embedding_model (str): The embedding model to use ("all-minilm", "nomic-embed-text", or "mxbai-embed-large").
            preprocess (bool): If True, preprocess the text before generating embeddings. Defaults to False.
        """
        self.db_type = db_type.lower()
        self.embedding_model = embedding_model
        self.textProcessor = TextProcess(preprocess=preprocess)

        if self.db_type == "chroma":
            # Initialize Chroma client
            self.client = chromadb.HttpClient(
                settings=Settings(allow_reset=True),
                host="localhost",
                port=8000
            )
            self.collection_name = COLLECTION_NAME
            self.collection = None

        elif self.db_type == "qdrant":
            # Initialize Qdrant client
            self.client = QdrantClient(url="http://localhost:6333")
            self.collection_name = COLLECTION_NAME

        elif self.db_type == "redis":
            # Initialize Redis client
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
            self.INDEX_NAME = "embedding_index"
            self.DOC_PREFIX = "doc:"
            self.DISTANCE_METRIC = "COSINE"
            self.VECTOR_DIM = self.get_vector_dim()
        else:
            raise ValueError("Unsupported database type. Use 'chroma', 'qdrant', or 'redis'.")

        self.clear_collection()

    def get_vector_dim(self):
        """
        Determines the vector dimension based on the embedding model.
        """
        model = self.embedding_model.lower()
        if model == "all-minilm":
            return 384
        elif model == "nomic-embed-text":
            return 768
        elif model == "mxbai-embed-large":
            return 1024
        else:
            raise ValueError("Unsupported embedding type. Use 'all-minilm', 'nomic-embed-text', or 'mxbai-embed-large'.")

    def clear_collection(self):
        """
        Clears the collection or index from the selected database.
        """
        if self.db_type == "chroma":
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            print("Chroma collection cleared.")

        elif self.db_type == "qdrant":
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception as e:
                print(f"Error deleting Qdrant collection: {e}")
            print("Qdrant collection cleared.")

        elif self.db_type == "redis":
            self.redis_client.flushdb()
            print("Redis store cleared.")

    def create_collection(self):
        """
        Creates (or recreates) the collection or index in the selected database.
        The vector dimension is set based on the embedding model.
        """
        if self.db_type == "chroma":
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.create_collection(name=COLLECTION_NAME)
            print("Chroma collection created.")

        elif self.db_type == "qdrant":
            dim = self.get_vector_dim()
            try:
                self.client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                print(f"Qdrant collection '{COLLECTION_NAME}' created with dimension {dim}.")
            except Exception as e:
                print(f"Error creating Qdrant collection: {e}")

        elif self.db_type == "redis":
            try:
                self.redis_client.execute_command(f"FT.DROPINDEX {self.INDEX_NAME} DD")
            except redis.exceptions.ResponseError:
                pass
            self.redis_client.execute_command(
                f"FT.CREATE {self.INDEX_NAME} ON HASH PREFIX 1 {self.DOC_PREFIX} "
                f"SCHEMA text TEXT embedding VECTOR HNSW 6 DIM {self.VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {self.DISTANCE_METRIC}"
            )
            print("Redis index created successfully.")

    def process_pdfs(self, data_dir: str, chunk_size=300, overlap=50):
        """
        Processes all PDFs in the given directory, extracts text, splits into chunks,
        generates embeddings using the specified embedding model, and stores them.

        Args:
            data_dir (str): Directory containing PDF files.
            chunk_size (int): Size of each text chunk.
            overlap (int): Overlap between chunks.
        """
        self.clear_collection()
        self.create_collection()

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = self.textProcessor.extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = self.textProcessor.split_text_into_chunks(text, chunk_size, overlap)
                    for chunk in chunks:
                        # Pass the embedding model so that the vector dimension is handled appropriately.
                        embedding = self.textProcessor.get_embedding(chunk, model=self.embedding_model)
                        self.store_embedding(file_name, page_num, chunk, embedding)
                print(f"Processed: {file_name}")

    def store_embedding(self, file_name: str, page_num: int, chunk: str, embedding: list):
        """
        Stores an embedding with metadata in the selected database.

        Args:
            file_name (str): PDF file name.
            page_num (int): Page number.
            chunk (str): Text chunk.
            embedding (list): Embedding vector.
        """
        if self.db_type == "chroma":
            chunk_id = f"{file_name}_page_{page_num}_chunk_{hash(chunk)}"
            self.collection.add(
                documents=[f"{file_name}_page_{page_num}_chunk_{chunk}"],
                embeddings=[embedding],
                ids=[chunk_id]
            )
            print(f"Stored embedding in Chroma for: {chunk}")

        elif self.db_type == "qdrant":
            point_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "file": file_name,
                            "page": page_num,
                            "chunk": chunk,
                        },
                    )
                ],
            )
            print(f"Stored embedding in Qdrant for: {file_name}, page {page_num}")

        elif self.db_type == "redis":
            key = f"{self.DOC_PREFIX}:{file_name}_page_{page_num}_chunk_{hash(chunk)}"
            self.redis_client.hset(
                key,
                mapping={
                    "file": file_name,
                    "page": page_num,
                    "chunk": chunk,
                    "embedding": np.array(embedding, dtype=np.float32).tobytes()
                },
            )
            print(f"Stored embedding in Redis for: {chunk}")

    def query(self, query_text: str, top_k: int = 5):
        """
        Queries the selected database for similar embeddings.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to return.
        """
        embedding = self.textProcessor.get_embedding(query_text, model=self.embedding_model)

        if self.db_type == "chroma":
            results = self.collection.query(
                query_embeddings=[embedding],
                query_texts=[query_text],
                n_results=top_k,
            )
            print("Chroma query results:")
            print(results)

        elif self.db_type == "qdrant":
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k,
            )
            print("Qdrant query results:")
            for hit in results:
                print(f"Score: {hit.score:.4f}")
                print(f"File: {hit.payload.get('file', 'unknown')}")
                print(f"Page: {hit.payload.get('page', 'unknown')}")
                print(f"Text: {hit.payload.get('chunk', 'unknown')}\n")

        elif self.db_type == "redis":
            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "vector_distance", "file", "page", "chunk")
                .dialect(2)
            )
            res = self.redis_client.ft(self.INDEX_NAME).search(
                q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
            )
            print("Redis query results:")
            for doc in res.docs:
                print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def main(data_dir: str = "../data/"):
    # db_types: "chroma", "qdrant", or "redis"
    # embedding_models: "all-minilm", "nomic-embed-text", or "mxbai-embed-large"

    # ie,
    ingest = UnifiedIngest(db_type="chroma", embedding_model="all-minilm",preprocess=True)
    ingest.process_pdfs(data_dir, chunk_size=300, overlap=50)

    print("\n---Testing Query---")
    ingest.query("name one vector database", top_k=5)


if __name__ == "__main__":
    main()