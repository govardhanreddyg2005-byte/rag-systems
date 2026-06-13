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

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        temperature=0
    )

    vector_db = FAISS.from_texts(chunks,embeddings)

    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_db.similarity_search(user_query)

        llm = ChatOllama(
            model="llama3",
            temperature=0
        )

        customized_prompt = ChatPromptTemplate.from_messages([
            """
            You are my assistant tutor.Answer the question based on the provided
            context and if you did not get the context simply say "I Don't have enough
            information to generate answer."
            {context}

            Question:
            {input}

            """
        ])

        stuff_chain = create_stuff_documents_chain(llm, customized_prompt)

        response = stuff_chain.invoke({"input":user_query,"context":matching_chunks})
        st.write(response)

