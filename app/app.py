import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "/vector_store"
MODEL_NAME = "llama3-8b-8192"

# --- UI SETUP ---
st.set_page_config(page_title="FinBot: Chat with Financials", page_icon="🤖", layout="centered")
st.title("🤖 FinBot: Chat with Financial Reports")
st.markdown("Ask me anything about the company's 10-K annual reports!")

# --- LOAD RESOURCES (CACHED) ---
@st.cache_resource
def load_embeddings():
    """Load the sentence transformer embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_vector_store(_embeddings):
    """Load the FAISS vector store."""
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        st.error("Vector store not found. Please run the ingestion process first.")
        return None
    return FAISS.load_local(VECTOR_STORE_PATH, _embeddings, allow_dangerous_deserialization=True)

embeddings = load_embeddings()
db = load_vector_store(embeddings)

@st.cache_resource
def load_llm():
    """Load the Groq LLM."""
    try:
        return ChatGroq(
            temperature=0,
            model_name=MODEL_NAME,
            api_key=os.environ.get("GROQ_API_KEY")
        )
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

llm = load_llm()

# --- PROMPT TEMPLATE ---
prompt_template = """
You are an expert financial analyst assistant. Use the following pieces of context from the company's annual report to answer the user's question. Your answer should be concise and directly based on the provided text.

If you don't know the answer from the context provided, just say that you don't know. Do not make up information.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- RAG CHAIN ---
if db and llm:
    retriever = db.as_retriever(search_kwargs={'k': 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
else:
    qa_chain = None

# --- INTERACTION ---
query = st.text_input("Your question:", placeholder="e.g., What were the total revenues last year?")

if st.button("Get Answer") and query:
    if not qa_chain:
        st.warning("The application is not fully configured. Please check the logs.")
    else:
        with st.spinner("Analyzing documents..."):
            try:
                result = qa_chain.invoke({"query": query})
                st.subheader("Answer")
                st.write(result["result"])

                with st.expander("Show Sources"):
                    st.write("The following sources were used to generate the answer:")
                    for doc in result["source_documents"]:
                        st.info(f"**Source:** (Page {doc.metadata.get('page', 'N/A')})")
                        st.text(doc.page_content)
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")