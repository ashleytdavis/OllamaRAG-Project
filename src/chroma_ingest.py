import os
import chromadb

from text_process import TextProcess
from chromadb.config import Settings

'''
    More information about everything I built here can be found here:
    https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
    I pretty much followed this guide verbatim
    
    https://docs.trychroma.com/docs/overview/getting-started
    this was also really helpful
'''

COLLECTION_NAME = "pdf_embeddings"

class ChromaDBIngest:
    '''
    A class that supports functionality of chromadb queries and ingestion. 
    '''
    def __init__(self):
        '''
        Intializes a connection to chromadb on port 8000
        '''
        self.textProcessor = TextProcess()
        # Initializing a chroma connection
        self.client = chromadb.HttpClient(
            settings=Settings(allow_reset=True),
            host="localhost",
            port=8000
        )
    

    def clear_chroma_collection(self, client):
        '''
        Deletes the specified chromaDB collection, effectively clearing it.
        
        Args:
            client: the chromadb client instance 
        '''
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        print("Chroma collection cleared")
    
    
    def create_chroma_collection(self, client):
        '''
        Deletes the specified collection (if already exists) and recreates it.
        
        Args:
            client: the chromadb client instance
        '''
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


    def process_pdfs(self, data_dir, client, chunk_size, overlap):
        '''
        Process all PDF files in a given directory. Extracts text, splits it into chunks,
        generates embeddings, and stores them.

        Args:
            data_dir (str): The directory containing PDF files to process.
        '''
        collection = self.create_chroma_collection(client)
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = self.textProcessor.extract_text_from_pdf(pdf_path)
                for page_num, text in text_by_page:
                    chunks = self.textProcessor.split_text_into_chunks(text, chunk_size, overlap)
                    for chunk_index, chunk in enumerate(chunks):
                        embedding = self.textProcessor.get_embedding(chunk)
                        self.store_embedding(
                            collection,
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk),
                            embedding=embedding,
                        )
                print(f" -----> Processed {file_name}")


    def store_embedding(self, collection, file: str, page: str, chunk: str, embedding: list):
        '''
        Stores an embedding in the specified ChromaDB collection.
        
        Args:
            collection: The ChromaDB collection object.
            file (str): The name of the PDF file.
            page (str): The page number.
            chunk (str): The text chunk.
            embedding (list): The vector embedding.
        '''
        chunk_id = f"{file}_page_{page}_chunk_{hash(chunk)}" # this was a reccomendation for unique IDs I found online
        collection.add( 
            documents=[f"{file}_page_{page}_chunk_{chunk}"],
            embeddings=[embedding],
            ids=[chunk_id]
        )
        print(f"Stored embedding for: {chunk}")


    def query_chroma(self, collection, query_text: str):
        '''
        Queries the specified ChromaDB collection using a text query.
        
        Args:
            collection: The ChromaDB collection object.
            query_text (str): The text to query with.
        '''
        query_embedding = self.textProcessor.get_embedding(query_text)
        results = collection.query(
            query_embeddings=[query_embedding],
            query_texts=[query_text],
            n_results=5,
        )
        print(results)


def main(data):
    chroma = ChromaDBIngest()
    client = chroma.client

    chroma.clear_chroma_collection(client)
    chroma.process_pdfs(data, client, 100, 50)
    print("\n\n\n\n---Done processing PDFs---\n\n\n\n")
    
    collection = client.get_collection(COLLECTION_NAME)
    chroma.query_chroma(collection, "name one vector databases")


if __name__ == "__main__":
    main("data/")
