# ClauseIQ – AI-Powered Insurance Policy Reasoning System


ClauseIQ is a **Retrieval-Augmented Generation (RAG)** application that analyzes insurance policy documents and provides **clause-grounded answers** to user queries. Instead of relying on general AI knowledge, it retrieves the most relevant policy clauses using semantic search and generates responses backed by the policy document.

---

## Features

* Extracts text from insurance policy PDFs using **PyMuPDF**
* Splits documents into clause-aware chunks for better retrieval
* Generates semantic embeddings using **Sentence Transformers**
* Stores and retrieves clauses with **FAISS Vector Database**
* Uses Gemini API for policy-grounded reasoning
* Interactive **Streamlit** interface for real-time query evaluation

---

## How It Works

1. Load an insurance policy PDF.
2. Extract text using PyMuPDF.
3. Split the document into clause-based chunks.
4. Generate embeddings with Sentence Transformers.
5. Store vectors in FAISS.
6. Retrieve the most relevant clauses for a user query.
7. Generate a grounded response using Gemini 2.5 Flash.

---

## Tech Stack

| Layer           | Technology            |
| --------------- | --------------------- |
| Language        | Python                |
| Frontend        | Streamlit             |
| AI Model        | Gemini 2.5 Flash      |
| Framework       | LangChain             |
| Embeddings      | Sentence Transformers |
| Vector Database | FAISS                 |
| PDF Processing  | PyMuPDF               |
| Environment     | python-dotenv         |

---

## Project Structure

<escape>CLAUSE-IQ/
│── app.py
│── rag_pipeline.py
│── embeddings.py
│── requirements.txt
│── .gitignore
│── .env.example
│── README.md
├── data/
│   ├── policy.pdf
│   └── .gitkeep</escape>

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CLAUSE-IQ.git
cd CLAUSE-IQ
```

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Query

> *"Does this policy cover hospitalization expenses?"*

The application retrieves the most relevant clauses from the policy document and generates a response that includes:

* Coverage Status
* Reasoning
* Relevant Policy Clauses
* Confidence Level

---

## Future Enhancements

* Support multiple policy documents
* Highlight retrieved clauses in the UI
* Conversation history
* Docker deployment
* Advanced policy comparison

---



