import os
import re
import fitz

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from embeddings import SentenceTransformerEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -----------------------------
# Load PDF using PyMuPDF
# -----------------------------

def load_pdf(pdf_path):

    document = fitz.open(pdf_path)

    full_text = ""

    for page in document:

        full_text += page.get_text("text")

    return full_text


# -----------------------------
# Split into clauses
# -----------------------------

def split_by_clause(text):

    clause_pattern = re.compile(
        r"(?=\n\s*(?:Section\s+)?(?:Clause\s+)?\d+(?:\.\d+)*[\.\):]\s)",
        re.MULTILINE
    )

    chunks = [
        chunk.strip()
        for chunk in clause_pattern.split(text)
        if chunk.strip()
    ]

    docs = []

    for chunk in chunks:

        docs.append(
            Document(
                page_content=chunk
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


# -----------------------------
# Build vector store
# -----------------------------

def build_vector_store():

    pdf_path = "data/policy.pdf"

    text = load_pdf(pdf_path)

    docs = split_by_clause(text)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local(
        "faiss_index"
    )

    return vectorstore


# -----------------------------
# Load vector store
# -----------------------------

def load_vector_store():

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-mpnet-base-v2"
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# -----------------------------
# Gemini model
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# -----------------------------
# Prompt template
# -----------------------------

prompt = PromptTemplate(

    template="""
You are an insurance expert.

Answer ONLY using the policy clauses below.

Give:

1. Coverage status
2. Reason
3. Relevant clause
4. Confidence

Context:

{context}

Question:

{question}

Answer:
""",

    input_variables=[
        "context",
        "question"
    ]
)


# -----------------------------
# Create chain
# -----------------------------

def get_chain():

    if os.path.exists("faiss_index"):

        vectorstore = load_vector_store()

    else:

        vectorstore = build_vector_store()

    retriever = vectorstore.as_retriever(

        search_kwargs={
            "k": 4
        }
    )

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=retriever,

        chain_type_kwargs={
            "prompt": prompt
        },

        return_source_documents=True
    )

    return qa_chain