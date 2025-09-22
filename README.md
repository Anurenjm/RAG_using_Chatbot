RAG Chatbot with FAISS and Google Gemini
Project Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload documents (PDF, TXT, etc.) and ask questions about their content. The system retrieves relevant information from the uploaded documents using FAISS as the vector database and generates precise, context-aware answers using Google Gemini LLM.
Unlike a normal chatbot that only relies on pre-trained knowledge, this system understands and answers questions based on custom documents uploaded by the user.
Project Architecture

Document Upload – Users upload files through a Streamlit web app.

Document Processing – Text is extracted and split into smaller chunks for better embedding and retrieval.

Embeddings – Each chunk is converted into numerical vectors using GoogleGenerativeAIEmbeddings.

Vector Store (FAISS) – Vectors are stored in FAISS, which enables fast similarity search.

Retriever – When the user asks a question, the retriever finds the most relevant chunks from FAISS.

LLM (Google Gemini) – The retrieved chunks are passed to Google Gemini, which generates the final human-like response.
Project Structure
rag_chatbot/
│── app.py                
│── config.py             
│
├── utils/
│   └── file_utils.py      
│
├── modules/
│   ├── loader.py          
│   ├── splitter.py       
│   ├── embeddings.py      
│   ├── vectorstore.py    
│   ├── retriever.py      
│   └── chat.py           
│
└── requirements.txt       
