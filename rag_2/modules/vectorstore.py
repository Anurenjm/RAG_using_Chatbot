import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from modules.embeddings import get_embeddings
from config import FAISS_DIR

def build_faiss_index(documents, persist_path=FAISS_DIR):
    """
    Build a FAISS vectorstore from a list of LangChain Document objects.
    Saves index to persist_path.
    """
    embeddings = get_embeddings()
    index = FAISS.from_documents(documents, embedding=embeddings)
    os.makedirs(persist_path, exist_ok=True)
    index.save_local(persist_path)
    return index
persist_path = "faiss_index"
def load_faiss_index():
    embeddings = get_embeddings()
    if os.path.exists(persist_path):
        return FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True  
        )
    else:
        return None

def add_documents_to_index(index, documents, persist_path=FAISS_DIR):
    """
    Add new docs to an existing index and persist.
    """
    embeddings = get_embeddings()
    index.add_documents(documents, embedding=embeddings)
    index.save_local(persist_path)
    return index
