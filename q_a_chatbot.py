import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    #Creating a vector store and storing as embeddings
    db = FAISS.from_texts(chunks,embedding_model)

    user_query = st.text_input("Type your query here")


    if user_query:
        matching_chunks=db.similarity_search(user_query)

        llm = ChatOllama(
            model="llama3",
            temperature=0
        )


        customized_prompt = ChatPromptTemplate.from_messages([
            """
            You are my assistant tutor.
            Answer the user question based on the following

            context:
            {context}

            Question:
            {input}
            
            """
        ])

        stuff_chain = create_stuff_documents_chain(
            llm, customized_prompt
        )

        response = stuff_chain.invoke({"input":user_query,"context":matching_chunks})
        st.write(response)

