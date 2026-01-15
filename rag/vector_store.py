# ============================================================================= #
# Project: Multimodal RAG Pipeline
# Develop by: Thiago Piovesan
# Description: Streamlit Main Interface
# Date: 2026-01-08 // YYYY-MM-DD
# Version: 0.1.0
# License: MIT
# ============================================================================= #
# Libs Importation:
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================= #
def vectorize_json(json_data):
    """
    Vectorizes the provided JSON data using FAISS and HuggingFace embeddings.

    Args:
        json_data (str): The JSON data as a string.

    Returns:
        FAISS: The FAISS vector store containing the embedded documents.
    """
    
    # --- 2. Chunk the Documents ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.create_documents([json_data])

    # --- 3. Generate Embeddings ---
    # Using an open-source model from Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- 4. Create a FAISS Vector Store ---
    # This step creates the index in-memory and stores the embeddings
    vector_store = FAISS.from_documents(docs, embeddings)
    print("FAISS index created and documents embedded.")

    # Optional: Save the index to disk for later reuse
    vector_store.save_local("faiss_rag_index")
    print("Index saved to 'faiss_rag_index'.")
    
# ============================================================================= #
    