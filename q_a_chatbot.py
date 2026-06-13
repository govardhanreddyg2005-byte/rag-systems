import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

st.header("mini-LLM")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf") 

#checking if file exists
if file is not None:
    
    #extracting the text from pdf file
    read_pdf = PdfReader(file)

    text = ""

    for page in read_pdf.pages:
        text+=page.extract_text()

    text_splitting = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."],
        chunk_size = 300,
        chunk_overlap = 50
    )

    chunks = text_splitting.split_text(text)
    

    