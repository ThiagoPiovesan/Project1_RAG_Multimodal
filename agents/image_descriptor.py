# ============================================================================= #
# Project: Multimodal RAG Pipeline
# Develop by: Thiago Piovesan
# Description: LLM Agent for Image Description
# Date: 2026-01-08 // YYYY-MM-DD
# Version: 0.1.0
# License: MIT
# ============================================================================= #
# Libs Importation:
import os
import time
import pytesseract
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_unstructured import UnstructuredLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract' 
# ============================================================================= #
# Get env variables
load_dotenv()  # Carrega as variáveis do arquivo .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada. Configure a variável de ambiente.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY não encontrada. Configure a variável de ambiente.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada. Configure a variável de ambiente.")
 
# ============================================================================= #
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, api_key=GOOGLE_API_KEY)

# ============================================================================= #
def describe_image(base64_image: str) -> str: 
    """
    Generate a description for an image provided in base64 format.

    Args:
        base64_image (str): The image in base64 format.

    Returns:
        str: The generated description.
    """
    
    # Remove o prefixo se já existir
    if base64_image.startswith('data:image'):
        base64_image = base64_image.split(',')[1]

    prompt_text = "Provide a detailed description of this image, focusing on main objects and colors for search indexing."
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    try:
        response = llm.invoke([message])
        print(response.content)
        # time.sleep(2)  # Aguarda 2 segundos entre chamadas
        return response.content
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("Quota excedida. Aguardando 60 segundos...")
            time.sleep(60)
            return describe_image(base64_image)  # Tenta novamente
        else:
            raise e
    
# ============================================================================= #