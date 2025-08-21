# 🤖 FinBot: Your AI Financial Report Analyst

FinBot is a fully containerized, end-to-end Retrieval-Augmented Generation (RAG) application that allows you to chat with the annual 10-K reports of any publicly traded company. Ask a question in plain English and get a fact-based answer sourced directly from the SEC filings.

## 📝 Project Overview

This project provides a complete, reusable blueprint for building a RAG-based chatbot. It addresses the real-world challenge of extracting specific insights from long, dense corporate documents. The key goals are accuracy, transparency (by showing sources), and ease of setup through Docker.

### 🚀 Key Features

* **Automated Data Sourcing:** Includes a script to download 10-K filings directly from the SEC EDGAR database for any stock ticker.
* **Dockerized Environment:** The entire application stack is managed with Docker Compose, ensuring a consistent and reproducible setup.
* **Vector-Based Retrieval:** Uses FAISS for lightning-fast, semantic retrieval of relevant financial information.
* **Fast LLM Generation:** Leverages the high-speed Groq API with Llama 3 for near-instant answers.
* **Source Verification:** Displays the exact text chunks used to generate an answer, ensuring faithfulness and user trust.
* **Clean, Interactive UI:** A simple and intuitive web interface built with Streamlit.

## ⚙️ Technology Stack

* **Orchestration:** Docker, Docker Compose
* **LLM:** Groq (Llama 3)
* **Knowledge Base:** FAISS (Vector Store)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Interface:** Streamlit
* **Data Processing:** LangChain, PyPDFLoader, SEC Edgar Downloader

## 🛠️ How to Set Up and Run Locally

Follow these steps to get the project running. **Docker is required.**

### Step 1: Clone the Repository

```bash
git clone [https://github.com/freillat/FinReportingLLM](https://github.com/freillat/FinReportingLLM)
cd FinReportingLLM
```


### Step 2: 🔑 API Key & Data Sourcing

This application relies on two key external services.
First, it uses the Groq API to power the Large Language Model (LLM), which provides incredibly fast and accurate answers. To use the app, you must obtain your own free Groq API Key from the Groq Console and place it in the .env file; this is how the application authenticates itself.

Rename the template `.env.example` to `.env` (or create a new `.env` file) and add your Groq API key:

```ini
# .env
GROQ_API_KEY="gsk_YourSecretGroqApiKeyGoesHere"
```

Second, the financial documents are sourced using the sec-edgar-downloader Python library. This tool automates fetching official 10-K filings from the SEC EDGAR database. To run the downloader script locally before building the Docker image, you must install this library with the following command:

Bash

pip install sec-edgar-downloader
The downloader script (edgar_downloader.py) also requires a user agent (e.g., your name and email) to make requests to the SEC's servers.


### Step 3: Download Financial Filings

Run the `edgar_downloader.py` script to fetch the 10-K reports. This script downloads the data into the `./data/` directory, which will be used by the Docker container.

By default, it downloads for MicroStrategy (`MSTR`). You can change this by passing arguments:

```bash
# Download for the default ticker (MSTR) for the last 5 years
python ingestion/edgar_downloader.py

# --- OR ---

# Download for a different ticker (e.g., NVIDIA for 3 years)
python ingestion/edgar_downloader.py --ticker NVDA --years 3
```
Wait for the download to complete before proceeding.

### Step 4: Build and Run the Docker Containers

Use Docker Compose to build the images and launch the services. The `ingestion` service will run first to create the vector store, and then the `app` service will start.

```bash
docker-compose up --build
```

Once the build is complete and the services are running, open your web browser and navigate to:

**`http://localhost:8501`**

You can now start asking questions to FinBot!

---
## 📊 Project Evaluation

The performance of this RAG application can be evaluated on three core metrics:

1.  **Faithfulness:** Measures if the generated answer is factually grounded in the retrieved context. This is the most critical metric for a financial chatbot.
2.  **Answer Relevancy:** Assesses if the answer directly addresses the user's question.
3.  **Context Precision:** Evaluates whether the retrieved text chunks were relevant for answering the question.

These metrics can be systematically measured by creating a small, hand-crafted Q&A dataset and using a framework like RAGAs or DeepEval.