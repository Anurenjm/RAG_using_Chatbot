import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL

def get_embeddings():
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
