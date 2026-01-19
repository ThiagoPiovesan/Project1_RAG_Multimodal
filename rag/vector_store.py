# ============================================================================= #
# Project: Multimodal RAG Pipeline
# Develop by: Thiago Piovesan
# Description: Streamlit Main Interface
# Date: 2026-01-08 // YYYY-MM-DD
# Version: 0.1.0
# License: MIT
# ============================================================================= #
# Libs Importation:
import os
import json

# from memchunk import Chunker
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================= #
def create_document(json_data: json, base_file_name: str) -> list[Document]:
    """
    Creates a document from the provided JSON data.
    This is a placeholder function and should be implemented based on specific requirements.

    Args:
        json_data (str): The JSON data as a string.
    """
    # Placeholder implementation
    print("Document created from JSON data.")

    docs = []
    for item in json_data:
        # O 'page_content' é o que o FAISS vai usar para gerar o embedding
        content = item["text"] 
        
        # Os metadados ajudam a LLM a citar fontes e você a depurar
        metadata = {
            "source": base_file_name,
            "page": item["metadata"].get("page_number"),
            "type": item["type"] # Ex: 'Table', 'Image', 'NarrativeText'
        }
        
        docs.append(Document(page_content=content, metadata=metadata))
        
    return docs

# ============================================================================= #
def vectorize_json(json_data: json, base_file_name: str) -> FAISS:
    """
    Vectorizes the provided JSON data using FAISS and HuggingFace embeddings.

    Args:
        json_data (str): The JSON data as a string.

    Returns:
        FAISS: The FAISS vector store containing the embedded documents.
    """
    
    # 1. Cria os documentos novos a partir do JSON atual
    new_docs = create_document(json_data, base_file_name)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "rag/faiss_rag_index"

    # ============================================================================= #
    # 2. Verifica se o índice já existe
    if os.path.exists(index_path) and os.path.exists(f"{index_path}/index.faiss"):
        print(f"🔄 Carregando índice existente em '{index_path}'...")
        
        # Carrega o índice existente (permitindo a desserialização perigosa se for local e seguro)
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
    # ============================================================================= #
        # Verifica se o documento já foi indexado
        existing_sources = set()
        if vector_store.docstore._dict: # Verifica se há documentos
            for doc_id, doc in vector_store.docstore._dict.items():
                if "source" in doc.metadata:
                    existing_sources.add(doc.metadata["source"])

        if base_file_name in existing_sources:
            print(f"⚠️ O arquivo '{base_file_name}' já está no índice. Pulando processamento.")
            return vector_store
        
    # ============================================================================= #
        # ADICIONA os novos documentos ao índice carregado
        vector_store.add_documents(new_docs)
        print(f"➕ Adicionados {len(new_docs)} novos chunks ao índice.")
        
    else:
        print("🆕 Criando um novo índice do zero...")
        # Cria o índice pela primeira vez
        vector_store = FAISS.from_documents(new_docs, embeddings)

    # 3. Salva o índice atualizado (sobrescrevendo a pasta com a versão combinada)
    vector_store.save_local(index_path)
    print("✅ Índice atualizado salvo com sucesso.")
    
    return vector_store
    
# ============================================================================= #
def load_vector_store() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # IMPORTANTE: allow_dangerous_deserialization=True
    vector_store = FAISS.load_local(
        "rag/faiss_rag_index", 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    return vector_store

def get_retriever_tool(vector_store: FAISS):
    """
    Factory function to create the tool with the vector_store pre-loaded.
    """
    
    @tool(response_format="content_and_artifact")
    def search_knowledge_base(query: str):
        """
        Call this tool to search for technical documents, pdfs, images and tables.
        Always use this tool to answer questions about the user's files.
        """
        # 1. Cria o objeto retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 2. EXECUTA a busca (faltava o invoke!)
        docs = retriever.invoke(query)
        
        # 3. Serializa para a LLM ler
        serialized = "\n\n".join(
            (f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page_number')})\nContent: {doc.page_content}")
            for doc in docs
        )
        
        # Retorna o texto para a LLM e os docs originais como artefato (opcional)
        return serialized, docs

    return search_knowledge_base