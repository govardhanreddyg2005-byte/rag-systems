# Project Title
Basic Question and Answer ChatBot(Q&A).

## 📌 Overview
Build a basic chatbot that generates the final answer based on user query from provided documents retrieving strong response acting as a Q&A chatbot
Utilizes Ollama local base model llama3 which is strong at semantic search and integrates the model with better streamlit UI user interface.

## 🚀 Key Features
*   **Local LLM Integration:** Powered by local `LLaMA` models llama3.
*   **Document Ingestion:** Automated text splitting and chunking using LangChain's `RecursiveCharacterTextSplitter`.
*   **Vector Database:** Local context retrieval optimized via `FAISS`.
*   **Interactive UI:** Real-time conversational chat interface built entirely with Streamlit.

## 🛠️ Architecture Workflow
1. **Data Ingestion:** Source pdf files are parsed from the local data directory.
2. **Chunking & Embedding:** Text chunks are transformed into vector embeddings using `Nomic-Embed-Text`.
3. **Storage:** Embeddings are persisted locally in a vector database index.
4. **Retrieval & Generation:** User queries fetch relevant context to augment the LLM system prompt for localized inference.

5. ## 💻 Getting Started
6. ### Prerequisites
* Python 3.11+
* Git

* ### Installation & Setup
1. **Clone the repository:**
   ```git clone https://github.com/govardhanreddyg2005-byte/rag-systems.git```
   ```cd rag-projects/chatbot```
   
2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
   
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4.**Run in Streamlit**
# USE following command to get UI interface experience
```streamlit run q_a_chatbot.py```

## 📂 Final Project Structure
```text
├── .gitignore          # add untracked files
├── README.md           # Project description
├── q_a_chatbot.py      # Core Streamlit app orchestration, ingestion, and retrieval logic
└── requirements.txt    # project library dependencies
```
   

  


