import fitz
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import ollama


class TextProcess:
    '''
    A class for processing text data, including extracting text from PDFs,
    splitting text into chunks, preprocessing text, and generating embeddings.
    
    The below functions were provided to us in the starter code, and are
    abstracted to reduce code duplication.
    '''
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')

    
    def extract_text_from_pdf(self, pdf_path):
        '''
        Extract text from a PDF file.

        Args:
            pdf_path (str): The file path to the PDF.

        Returns:
            list: A list of tuples where each tuple contains the page number and the text content of that page.
        '''
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page


    def split_text_into_chunks(self, text, chunk_size=300, overlap=50):
        '''
        Split text into chunks of approximately `chunk_size` words with an overlap.

        Args:
            text (str): The input text to split.
            chunk_size (int): The number of words per chunk. Default is 300.
            overlap (int): The number of overlapping words between consecutive chunks. Default is 50.

        Returns:
            list: A list of text chunks.
        '''
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks


    def preprocess_text(self, text):
        '''
        Preprocess text by removing extra spaces, punctuation, and stopwords.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The cleaned and preprocessed text.
        '''
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(filtered_tokens)
                
                
    def get_embedding(self, text: str, model: str = "nomic-embed-text") -> list:
        '''
        Generate an embedding for the given text using the specified model.

        Args:
            text (str): The input text to generate an embedding for.
            model (str): The model to use for generating embeddings. Default is "nomic-embed-text".

        Returns:
            list: The embedding vector as a list of floats.
        '''
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    
    