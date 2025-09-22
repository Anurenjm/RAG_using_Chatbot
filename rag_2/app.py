import streamlit as st
import os
from config import FAISS_DIR
from utils.file_utils import save_uploaded_file
from modules.loader import load_file
from modules.splitter import split_documents
from modules.vectorstore import build_faiss_index, load_faiss_index, add_documents_to_index
from modules.chat import build_qa_chain, answer_question
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.vectorstore import build_faiss_index, load_faiss_index, add_documents_to_index
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



st.set_page_config(page_title="RAG Chat (FAISS + Gemini)", layout="centered")

st.title("RAG Chatbot — FAISS + Gemini")

# Check/load existing index
faiss_index = load_faiss_index()
if faiss_index:
    st.info("Loaded existing FAISS index.")
else:
    st.info("No FAISS index found. You can upload files to build one.")

# File upload
st.header("Upload files (PDF / TXT / CSV / DOCX)")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf","txt","csv","docx","doc","md"])

if uploaded_files:
    docs_all = []
    for upl in uploaded_files:
        tmp_path = save_uploaded_file(upl)
        try:
            docs = load_file(tmp_path)
            if docs:
                docs_all.extend(docs)
                st.success(f"Loaded {len(docs)} documents from {upl.name}")
            else:
                st.warning(f"No documents extracted from {upl.name} (unsupported format or empty).")
        except Exception as e:
            st.error(f"Failed to load {upl.name}: {e}")

    if docs_all:
       
        chunks = split_documents(docs_all)
        st.write(f"Created {len(chunks)} chunks from uploaded files.")

        # build or add to index
        if faiss_index is None:
            faiss_index = build_faiss_index(chunks)
            st.success("Built new FAISS index and persisted it.")
        else:
            faiss_index = add_documents_to_index(faiss_index, chunks)
            st.success("Added uploaded docs to existing FAISS index and persisted it.")


st.header("Ask something from your uploaded docs")
question = st.text_input("Ask a question (context will be retrieved from indexed files)")

if st.button("Get Answer"):
    if faiss_index is None:
        st.error("No FAISS index available. Upload files first.")
    elif not question:
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            qa_chain = build_qa_chain(faiss_index)
            answer = answer_question(qa_chain, question)
            st.subheader("Answer")
            st.write(answer)


if st.button("Delete FAISS index (reset)"):
    if os.path.exists(FAISS_DIR):
        import shutil
        shutil.rmtree(FAISS_DIR)
    faiss_index = None
    st.success("FAISS index removed. You can upload files to build a new one.")
