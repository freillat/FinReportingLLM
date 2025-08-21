import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
DATA_PATH = '/data'
VECTOR_STORE_PATH = '/vector_store'

def create_vector_store():
    """
    Loads PDF documents, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    # Find all PDF files in the SEC filings directory
    pdf_files = glob.glob(os.path.join(DATA_PATH, "sec-edgar-filings", "*", "10-K", "*", "full-submission.txt"))
    
    if not pdf_files:
        print("No 10-K full-submission.txt files found. Please run the downloader first.")
        # Let's adjust to also look for PDF filings directly for more general use
        pdf_files = glob.glob(os.path.join(DATA_PATH, "**/*.pdf"), recursive=True)
        if not pdf_files:
            print("No PDF files found in the data directory either.")
            return

    print(f"Found {len(pdf_files)} documents to process.")
    
    # Load documents
    loaders = [PyPDFLoader(file_path) for file_path in pdf_files]
    documents = []
    for loader in loaders:
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