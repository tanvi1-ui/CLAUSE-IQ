# CLAUSE-IQ
ClauseIQ – AI-Powered Insurance Policy Reasoning System - Retrieval-Augmented Generation (RAG) application that analyzes insurance policy documents and answers user queries using policy clauses rather than relying on general AI knowledge.

## Features

* PDF policy ingestion
* Clause-aware document splitting
* Semantic retrieval with FAISS
* SentenceTransformer embeddings
* Gemini 2.5 Flash reasoning
* Streamlit web interface


### Workflow

1. Upload policy PDF.
2. Extract text using PyMuPDF.
3. Split into clause-based chunks.
4. Generate embeddings.
5. Store vectors in FAISS.
6. Retrieve relevant clauses.
7. Gemini generates a grounded answer.

## Tech Stack

| Layer          | Technology           |
| -------------- | -------------------- |
| Frontend       | Streamlit            |
| Backend        | Python               |
| AI             | Gemini 2.5 Flash     |
| Framework      | LangChain            |
| Embeddings     | SentenceTransformers |
| Vector DB      | FAISS                |
| PDF Processing | PyMuPDF              |


## Project Structure

ClauseIQ/
│── app.py
│── rag_pipeline.py
│── embeddings.py
│── requirements.txt
│── .gitignore
│── .env.example
│── README.md
├── data/
├── assets/
└── faiss_index/</escape>

