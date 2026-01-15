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
import io
import time
import json
import base64
import hashlib
import streamlit as st

from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from agents.image_descriptor import describe_image

# ============================================================================= #
file_path = "./data"
base_file_name = "layout-parser-paper"

# ============================================================================= #
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_hash = hashlib.md5(image_file.read()).hexdigest()
    
    with Image.open(image_path) as img:    
        # Converte para RGB se necessário
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Salva em buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        
    return base64.b64encode(buffered.getvalue()).decode(), image_hash

# ============================================================================= #
def process_pdf_and_describe_images(file_path, base_file_name, elements, described_images_hashes) -> dict:
    """
    Process the PDF elements, describe images, and save output to JSON.

    Args:
        file_path (str): File path for saving output. 
        base_file_name (str): Base name for the output file.
        elements (list): List of elements extracted from the PDF.
        described_images_hashes (dict): Dictionary of image hashes and their descriptions.

    Returns:
        dict: Updated dictionary of image hashes and their descriptions.
    """
    
    for el in elements:
        if el.category == "Table":
            html_table = el.metadata.text_as_html
            el.text = f"Table HTML: {html_table}"
            # print(f"Tabela encontrada: {html_table}")   
            
        elif el.category == "Image":
            # Salva a imagem com um path no metadata
            image_path = el.metadata.image_path
            b64, image_hash = encode_image(image_path=image_path)
            
            if image_hash in described_images_hashes:
                description = described_images_hashes[image_hash]
                # print(f"Descrição da imagem (hash: {image_hash}) já existente: {description}")
            else:
                # Chama o agente de descrição de imagem
                description = describe_image(base64_image=b64)
                # image_descriptions.append({'hash': image_hash, 'description': description})
                described_images_hashes[image_hash] = description
                # print(f"Nova descrição da imagem (hash: {image_hash}): {description}")
            
            # Substitui o conteúdo do elemento:
            el.text = f"Image Description: {description}"            # Gere um hash da imagem para verificar se já foi descrita

    elements_to_json(elements=elements, filename=f"{file_path}/{base_file_name}-output.json")
    
    return described_images_hashes
# ============================================================================= #

def main():
    described_images_hashes = {}
    
# ============================================================================= #
    # Streamlit Interface:
    st.title("Multimodal RAG Pipeline - Image Description Agent")
    st.write("This application processes a PDF document, extracts images, and generates descriptions for each image using a language model.")
    
    st.markdown("---")
    
    # Aba 1: File Upload:
    st.header("1. PDF File Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Salva o arquivo temporariamente
        with open(f"{file_path}/{base_file_name}.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully!")
        
        # TODO: Adicionar multi-processing para processar vários documentos ao mesmo tempo
        # Botão para iniciar o processamento
        if st.button("Process PDF and Describe Images"):
            st.info("Processing the PDF file. This may take a few moments...")
            start_time = time.time()
            
            elements = partition_pdf(
                filename=f"{file_path}/{base_file_name}.pdf",
                strategy="hi_res",                                  # Obrigatório para tabelas e imagens
                infer_table_structure=True,                         # Extrai a estrutura da tabela
                extract_images_in_pdf=True,                         # Salva as imagens localmente
                extract_image_block_output_dir="./data/temp_images",# Pasta onde as imagens vão cair
                extract_image_block_types=["Image", "Table"]        # O que deve ser "recortado"
            )
            
            described_images_hashes = process_pdf_and_describe_images(file_path, base_file_name, elements, described_images_hashes)
            
            end_time = time.time()
            st.success(f"Processing completed in {end_time - start_time:.2f} seconds!")
            st.markdown(f"[Download Output JSON](./data/{base_file_name}-output.json)")
            
    
# ============================================================================= #
    # Aba 2: RAG Service (Futuro)
    st.markdown("---")
    st.header("2. RAG Service (Future Implementation)")
    st.write("This section will be implemented in the future to demonstrate the RAG service using the processed data.")
    
    # Perguntas e respostas com o agente RAG (Futuro)
    st.text_area("Ask a question about the document:", height=100)
    st.button("Get Answer")

# ============================================================================= #
if __name__ == "__main__":
    main()
# from services.rag_service import RAGService