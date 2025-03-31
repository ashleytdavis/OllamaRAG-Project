import os
from qdrant_client import QdrantClient
from text_process import TextProcess
from qdrant_client.http import models
from typing import List
import uuid

COLLECTION_NAME = "pdf_embeddings"

class QdrantDBIngest:
    '''
    A class that supports functionality of chromadb queries and ingestion. 
    '''
    def __init__(self):
        '''
        Intializes a connection to chromadb on port 8000
        '''
        self.textProcessor = TextProcess()
        # Initializing a qdrant connection
        self.client = QdrantClient(url="http://localhost:6333")
        
    

    def clear_qdrant_collection(self, client):
        '''
        Deletes the specified chromaDB collection, effectively clearing it.
        
        Args:
            client: the chromadb client instance 
        '''
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        print("Qdrant collection cleared")
    
    
    def create_qdrant_collection(self, client):
        '''
        Deletes the specified collection (if already exists) and recreates it.
        
        Args:
            client: the chromadb client instance
        '''
        # delete the collection if it already exists
        try:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,  # Adjust based on your embedding model (e.g., 384 for all-MiniLM-L6-v2)
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Collection '{COLLECTION_NAME}' created.")
        except Exception as e:
            print(f"Error creating collection: {e}")


    def process_pdfs(self, data_dir: str):
        '''
        Processes all PDFs in `data_dir`, extracts text, splits into chunks, and stores embeddings in Qdrant.
        '''
        self.clear_qdrant_collection(self.client)
        self.create_qdrant_collection(self.client)

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = self.textProcessor.extract_text_from_pdf(pdf_path)
                
                for page_num, text in text_by_page:
                    chunks = self.textProcessor.split_text_into_chunks(text)
                    
                    for chunk_index, chunk in enumerate(chunks):
                        embedding = self.textProcessor.get_embedding(chunk)
                        self.store_embedding(
                            file_name=file_name,
                            page_num=page_num,
                            chunk_text=chunk,
                            embedding=embedding,
                        )
                print(f"Processed: {file_name}")


    def store_embedding(
        self,
        file_name: str,
        page_num: int,
        chunk_text: str,
        embedding: List[float],
    ):
        '''
        Stores an embedding in Qdrant with metadata.
        '''
        point_id = str(uuid.uuid4())  # Simple unique ID
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file": file_name,
                        "page": page_num,
                        "chunk": chunk_text,
                    },
                )
            ],
        )
        print(f"Stored embedding for: {file_name}, page {page_num}")


    def query_qdrant(self, query_text: str, top_k: int = 5):
        '''
        Queries Qdrant for similar embeddings.
        '''
        query_embedding = self.textProcessor.get_embedding(query_text)
        
        results = self.client.query_points(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        )
        
        print("\nSearch Results:")
        for hit in results:
            print(f"Score: {hit.score:.4f}")
            print(f"File: {hit.payload['file']}")
            print(f"Page: {hit.payload['page']}")
            print(f"Text: {hit.payload['chunk']}\n")

def main(data_dir: str = "data/"):
    qdrant = QdrantDBIngest()
    qdrant.process_pdfs(data_dir)
    
    print("\n---Testing Query---")
    qdrant.query_qdrant("name one vector database")

if __name__ == "__main__":
    main()