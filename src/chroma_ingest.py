import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

from text_process import TextProcess
'''
    More information about everything I built here can be found here:
    https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
    I pretty much followed this guide verbatim
    
    https://docs.trychroma.com/docs/overview/getting-started
    this was also really helpful
'''
import chromadb
from chromadb.config import Settings


nltk.download('punkt')
nltk.download('stopwords')

textProcessor = TextProcess()


VECTOR_DIM = 768
COLLECTION_NAME = "pdf_embeddings"

    

def clear_chroma_collection(client):
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    print("Chroma collection cleared")
    
def create_chroma_collection(client):
    # delete the collection if it already exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME
    )
    print("Chroma collcetion created")
    return collection


def process_pdfs(data_dir, client):
        '''
        Process all PDF files in a given directory. Extracts text, splits it into chunks,
        generates embeddings, and stores them.

        Args:
            data_dir (str): The directory containing PDF files to process.
        '''
        collection = create_chroma_collection(client)
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = textProcessor.extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = textProcessor.split_text_into_chunks(text)
                    for chunk_index, chunk in enumerate(chunks):
                        embedding = textProcessor.get_embedding(chunk)
                        store_embedding(
                            collection,
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk),
                            embedding=embedding,
                        )
                print(f" -----> Processed {file_name}")

# store the embedding in chroma
def store_embedding(collection, file: str, page: str, chunk: str, embedding: list):
    chunk_id = f"{file}_page_{page}_chunk_{hash(chunk)}" # this was a reccomendation for unique IDs I found online
    collection.add( 
        documents=[f"{file}_page_{page}_chunk_{chunk}"],
        embeddings=[embedding],
        ids=[chunk_id]
    )
    print(f"Stored embedding for: {chunk}")


# Query ChromaDB
def query_chroma(collection, query_text: str):
    query_embedding = textProcessor.get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )
    for result in results:
        print(f"{result.id} \n ----> {result.distance}\n")


def main(data):
  
    client = chromadb.HttpClient(
        settings=Settings(allow_reset=True),
        host="localhost", 
        port=8000)
    '''
    clear_chroma_collection(client)
    process_pdfs(data, client)
    print("\n\n\n\n---Done processing PDFs---\n\n\n\n")
    '''
    collection = client.get_collection(COLLECTION_NAME)
    query_chroma(collection, "Efficient search in vector databases")


if __name__ == "__main__":
    main("data/")
