import os
import glob
import time
import tqdm  # <-- Import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
DATA_PATH = 'data'
VECTOR_STORE_PATH = 'vector_store'

def create_vector_store():
    """
    Loads documents, splits them, creates embeddings, and stores them in FAISS.
    """
    start_time = time.time()
    print("--- Starting Ingestion Process ---")

    # 1. & 2. Find and Load Documents (same as before)
    print(f"1. Searching for documents in '{DATA_PATH}'...")
    all_files = glob.glob(os.path.join(DATA_PATH, "sec-edgar-filings", "**", "*.*"), recursive=True)
    if not all_files:
        print("Error: No files found.")
        return
    print(f"   Found {len(all_files)} files.")
    
    print("2. Loading documents...")
    documents = []
    for file_path in all_files:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f"   - Warning: Could not load file {os.path.basename(file_path)}. Error: {e}")
            continue
    print(f"   Loaded {len(documents)} document pages/sections.")

    # 3. Split Documents (same as before)
    print("3. Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"   Split into {len(docs)} chunks.")

    # 4. Load Embedding Model (same as before)
    print("4. Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("   Embedding model loaded.")

    # 5. Create FAISS vector store with a progress bar
    print("5. Creating FAISS vector store from chunks...")
    
    batch_size = 32  # Process documents in batches of 32
    db = None

    # Use tqdm to create a visual progress bar
    for i in tqdm.tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        if db is None:
            # Create the index from the first batch
            db = FAISS.from_documents(batch, embeddings)
        else:
            # Add subsequent batches to the existing index
            db.add_documents(batch)
            
    # Save the final index
    db.save_local(VECTOR_STORE_PATH)
    
    end_time = time.time()
    print(f"--- Vector store created successfully in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    create_vector_store()