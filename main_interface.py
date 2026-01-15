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
import json
import base64
import hashlib
import webbrowser 
import pytesseract
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
def main():
    described_images_hashes = {}
    
    elements = partition_pdf(
        filename=f"{file_path}/{base_file_name}.pdf",
        strategy="hi_res",                                  # Obrigatório para tabelas e imagens
        infer_table_structure=True,                         # Extrai a estrutura da tabela
        extract_images_in_pdf=True,                         # Salva as imagens localmente
        extract_image_block_output_dir="./data/temp_images",# Pasta onde as imagens vão cair
        extract_image_block_types=["Image", "Table"]        # O que deve ser "recortado"
                             )
    
# ============================================================================= #
    for el in elements:
        if el.category == "Table":
            html_table = el.metadata.text_as_html
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

# ============================================================================= #
if __name__ == "__main__":
    main()
# from services.rag_service import RAGService