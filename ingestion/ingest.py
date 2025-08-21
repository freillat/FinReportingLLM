import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
DATA_PATH = '/data'
VECTOR_STORE_PATH = '/vector_store'

def create_vector_store():
    """
    Loads documents from .txt and .pdf files, splits them into chunks,
    creates embeddings, and stores them in a FAISS vector store.
    """
    print("Searching for documents to process...")
    # Find all relevant files in the SEC filings directory
    txt_files = glob.glob(os.path.join(DATA_PATH, "sec-edgar-filings", "**", "*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(DATA_PATH, "sec-edgar-filings", "**", "*.pdf"), recursive=True)
    
    all_files = txt_files + pdf_files
    
    if not all_files:
        print("No .txt or .pdf files found in the data directory. Please ensure the downloader has run successfully.")
        return

    print(f"Found {len(all_files)} documents to process.")
    
    # Load documents using the appropriate loader
    documents = []
    for file_path in all_files:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue # Skip other file types
        
        documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    print(f"Split documents into {len(docs)} chunks.")

    # Create embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save FAISS vector store
    print("Creating vector store...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    print("Vector store created and saved successfully.")

if __name__ == "__main__":
    create_vector_store()