from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL
from modules.retriever import get_retriever

def build_qa_chain(faiss_index, llm_model_name=LLM_MODEL, temperature=0.0):
    """
    Returns a RetrievalQA chain using Gemini Chat as the LLM.
    """
    retriever = get_retriever(faiss_index)
    llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=temperature)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def answer_question(qa_chain, question):
    """
    Run the QA chain and return the answer string.
    """
    return qa_chain.run(question)
