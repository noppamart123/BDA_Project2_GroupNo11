
import os
import re
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline


GROUP_NO = "GroupNo11"

GROUP_MEMBERS = [
    "6631501046 - THANAWAT JANHAN",
    "6631501050 - THARAKSA YASA",
    "6631501057 - TEERASAK JEEFU",
    "6631501059 - NACHANON SANGSURIYAN",
    "6631501060 - NOPPAMART PENGMANEE",
    "6631501076 - PANNAWIT KLUYPUK",
    "6631501077 - PATTAKORN JAIJIT",
    
]


DATA_PATH = "data"
PROJECT_NAME = f"BDA_Project2_{GROUP_NO}"
APP_TITLE = "MFU TOEIC RAG Chatbot"


st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="📘",
    layout="wide"
)

st.title(f"📘 {PROJECT_NAME}")

st.markdown(f"""
## Project Name: {APP_TITLE}

### Project Description
This project develops an AI chatbot using **Retrieval-Augmented Generation (RAG)**  
to answer TOEIC registration and test preparation questions from PDF and DOCX documents.

### AI Concepts Used
- Retrieval-Augmented Generation (RAG)
- Text Chunking
- Embedding Model
- FAISS Vector Database
- Retriever
- Local LLM
- Streamlit Web Application

### Group Members
""")

for member in GROUP_MEMBERS:
    st.markdown(f"- {member}")

st.write("---")


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(file_path):
    pages = []
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append({
                "text": clean_text(page_text),
                "source": os.path.basename(file_path),
                "page": page_num
            })

    return pages


def read_docx(file_path):
    doc = Document(file_path)
    text = ""

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text.strip() + "\n"

    if not text.strip():
        return []

    return [{
        "text": clean_text(text),
        "source": os.path.basename(file_path),
        "page": "-"
    }]


def load_documents():
    documents = []

    if not os.path.exists(DATA_PATH):
        return documents

    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)

        if filename.lower().endswith(".pdf"):
            documents.extend(read_pdf(file_path))

        elif filename.lower().endswith(".docx"):
            documents.extend(read_docx(file_path))

    return documents


def split_documents(documents, chunk_size=800, overlap=150):
    chunks = []

    for doc in documents:
        text = doc["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"]
                })

            start += chunk_size - overlap

    return chunks


@st.cache_resource
def build_rag_system():
    documents = load_documents()

    if len(documents) == 0:
        return None, None, None, None, None

    chunks = split_documents(documents)

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    local_llm = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=300
    )

    return chunks, embedding_model, index, local_llm, documents


def retrieve(question, chunks, embedding_model, index, top_k=4):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    retrieved_chunks = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


def generate_answer(question, retrieved_chunks, local_llm):
    context = "\n\n".join([
        f"Source: {chunk['source']} | Page: {chunk['page']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    prompt = f"""
You are a helpful Thai TOEIC assistant.

Answer the question in Thai only.
Use only the context below.
If the answer is not found in the context, answer:
"ไม่พบข้อมูลนี้ในเอกสารที่ให้มา"

Context:
{context}

Question:
{question}

Thai Answer:
"""

    result = local_llm(prompt)[0]["generated_text"]
    return result.strip()


chunks, embedding_model, index, local_llm, documents = build_rag_system()

if chunks is None:
    st.error("ไม่พบไฟล์ PDF หรือ DOCX ในโฟลเดอร์ data")
    st.stop()


with st.sidebar:
    st.header("📂 Dataset Information")

    st.write("Dataset Folder:")
    st.code(DATA_PATH)

    st.write("Files Found:")
    for filename in os.listdir(DATA_PATH):
        if filename.lower().endswith((".pdf", ".docx")):
            st.write(f"- {filename}")

    st.write("---")
    st.write(f"Total Documents: {len(documents) if documents is not None else 0}")
    st.write(f"Total Chunks: {len(chunks) if chunks is not None else 0}")

top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=8,
        value=4
    )


st.subheader("💬 Ask TOEIC Questions")

example_questions = [
    "ต้องไปถึงสนามสอบก่อนกี่นาที",
    "ต้องเตรียมเอกสารอะไรไปสอบ TOEIC",
    "นักศึกษา มฟล. เลือกสอบประเภทใดได้บ้าง",
    "ผลคะแนนสอบ TOEIC มีอายุกี่ปี",
    "ถ้าต้องการสอบซ้ำต้องเว้นกี่วัน",
    "ห้ามนำอะไรเข้าห้องสอบ TOEIC",
    "ต้องเตรียมรูปถ่ายไปเองไหม",
    "ต้องเตรียมอุปกรณ์เครื่องเขียนไหม",
    "ถ้าชำระเงินแล้วไม่ไปสอบ ขอเงินคืนได้ไหม"
]

selected_question = st.selectbox(
    "เลือกคำถามตัวอย่าง",
    [""] + example_questions
)

question = st.text_input(
    "หรือพิมพ์คำถามเอง",
    value=selected_question
)

if st.button("Ask AI"):
    if not question.strip():
        st.warning("กรุณาพิมพ์คำถามก่อน")
    else:
        with st.spinner("กำลังค้นหาข้อมูลจากเอกสารและสร้างคำตอบ..."):
            retrieved_chunks = retrieve(
                question,
                chunks,
                embedding_model,
                index,
                top_k=top_k
            )

            answer = generate_answer(
                question,
                retrieved_chunks,
                local_llm
            )

        st.subheader("✅ Answer")
        st.success(answer)

        st.subheader("🔎 Retrieved Sources")

        for i, chunk in enumerate(retrieved_chunks, start=1):
            with st.expander(
                f"Source {i}: {chunk['source']} | Page {chunk['page']}"
            ):
                st.write(chunk["text"])


st.write("---")
st.caption(
    "Built with Python, Streamlit, RAG, Sentence Transformers, FAISS, and Local LLM"
)
