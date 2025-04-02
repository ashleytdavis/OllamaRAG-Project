from search import InteractiveSearch
from UnifiedInjest import UnifiedIngest
import argparse



def main():
    #gather necessary inputs
    parser = argparse.ArgumentParser(description="Run an interactive search")
    parser.add_argument("path", type=str, help="Data path")
    args = parser.parse_args()
    db_types = ["redis", "chroma", "qdrant"]
    embed_types = ["all-minilm", "nomic-embed-text", "mxbai-embed-large"]
    rag_types = ["llama3.2", "mistral", "deepseek-r1"]
    while True:
        db_type = input("Enter the database type (redis/chroma/qdrant): ")
        embed_type = input("Enter the embedding model (all-minilm/nomic-embed-text/mxbai-embed-large): ")
        RAG_type =  input("Enter the RAG model (llama3.2/mistral/deepseek-r1): ")
        chunk = int(input("Enter the chunk size (optimal 500, must be integer): "))
        overlap = int(input("Enter the overlap size (optimal 50, must be integer): "))
        if db_type in db_types and embed_type in embed_types and RAG_type in rag_types:
            break
        else:
            print("Invalid input. Please try again.")
    # Prep the data
    ingest = UnifiedIngest(db_type=db_type, embedding_model=embed_type,preprocess=True)
    ingest.process_pdfs(args.path, chunk_size=chunk, overlap=overlap)
    # Initiate search
    InteractiveSearch(embed_type, db_type,  RAG_type)

    


if __name__ == "__main__":
    main()
    

    