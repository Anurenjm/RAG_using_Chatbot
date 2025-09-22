from config import TOP_K

def get_retriever(faiss_index, k=TOP_K):
    """
    Returns a retriever object for the index with search kwargs.
    """
    return faiss_index.as_retriever(search_kwargs={"k": k})
