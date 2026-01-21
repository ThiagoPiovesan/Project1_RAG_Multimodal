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
import zipfile
import hashlib
import pandas as pd
import streamlit as st

from PIL import Image
from rag.vector_store import vectorize_json
from agents.rag_agent import rag_agent_response
from agents.image_descriptor import describe_image
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# ============================================================================= #
file_path = "./data"
base_file_name = "layout-parser-paper"

# --- Configuração da Página ---
st.set_page_config(
    page_title="Sistema Multimodal RAG",
    page_icon="🤖",
    layout="wide"
)

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
def process_pdf(file_path, base_file_name, described_images_hashes, mock_file=None):
    start_time = time.time()
                
    if mock_file:
        with open(f"{file_path}/{base_file_name}.pdf", "wb") as f:
            f.write(mock_file.getvalue())
                
    elements = partition_pdf(
        filename=f"{file_path}/{base_file_name}.pdf",
        strategy="hi_res",                                  # Obrigatório para tabelas e imagens
        infer_table_structure=True,                         # Extrai a estrutura da tabela
        extract_images_in_pdf=True,                         # Salva as imagens localmente
        extract_image_block_output_dir="./data/temp_images",# Pasta onde as imagens vão cair
        extract_image_block_types=["Image", "Table"]        # O que deve ser "recortado"
    )
    
    described_images_hashes, elements = describe_images_and_tables(elements, described_images_hashes)
    
    elements_to_json(elements=elements, filename=f"{file_path}/{base_file_name}-output.json")
    
    end_time = time.time()
    st.success(f"Processing completed in {end_time - start_time:.2f} seconds!")
    
    # ---------------------------------------------------------------------------- #
    # Extract Embbeding to RAG:
    json_data = ""
    with open(f"{file_path}/{base_file_name}-output.json", "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    
    vectorize_json(json_data = json_data, base_file_name = base_file_name)
    
    return described_images_hashes

# ============================================================================= #
def describe_images_and_tables(elements, described_images_hashes) -> dict:
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
    
    # ---------------------------------------------------------------------------- #
    for el in elements:
        if el.category == "Table":
            html_table = el.metadata.text_as_html
            el.text = f"Table HTML: {html_table}"
            # print(f"Tabela encontrada: {html_table}")   
            
    # ---------------------------------------------------------------------------- #
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
    
    return described_images_hashes, elements

# ============================================================================= #
def extract_zip(uploaded_file, file_path, described_images_hashes):
    # Extrai o arquivo zip
    st.subheader("📦 Processamento de Arquivo ZIP")
            
    st.info("O sistema irá extrair e processar automaticamente todos os arquivos suportados dentro do ZIP.")
    
    # ---------------------------------------------------------------------------- #
    try:
        zip_buffer = io.BytesIO(uploaded_file.getvalue())
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            file_list = [f for f in zip_ref.namelist() 
                        if not f.endswith('/') and not f.startswith('__MACOSX')]
            
            st.write(f"**Arquivos encontrados:** {len(file_list)}")
            
            # Contar por tipo
            tipos_encontrados = {}
            for fname in file_list:
                ext = fname.split('.')[-1].lower()
                tipos_encontrados[ext] = tipos_encontrados.get(ext, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 PDF", tipos_encontrados.get('pdf', 0))
            with col2:
                st.metric("🖼️ Imagens", 
                            tipos_encontrados.get('png', 0) + 
                            tipos_encontrados.get('jpg', 0) + 
                            tipos_encontrados.get('jpeg', 0))
            with col3:
                st.metric("📊 Outros", sum(tipos_encontrados.values()) - 
                            tipos_encontrados.get('pdf', 0) - 
                            tipos_encontrados.get('png', 0) - 
                            tipos_encontrados.get('jpg', 0) - 
                            tipos_encontrados.get('jpeg', 0))
            
            with st.expander("📋 Ver lista de arquivos"):
                for fname in file_list:
                    st.write(f"- {fname}")
                    
    # ---------------------------------------------------------------------------- #
    except Exception as e:
        st.error(f"Erro ao ler ZIP: {e}")
    
    # ============================================================================= #
    # Botão para processar
    if st.button("🚀 Processar Todos os Arquivos do ZIP", width='stretch'):
        
        # Contadores
        processados = 0
        com_sucesso = 0
        com_erro = 0
        nao_suportados = 0
        
        resultados_detalhados = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    # ---------------------------------------------------------------------------- #
        try:
            zip_buffer = io.BytesIO(uploaded_file.getvalue())
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                file_list = [f for f in zip_ref.namelist() 
                            if not f.endswith('/') and not f.startswith('__MACOSX')]
                
                total_files = len(file_list)
                
                for idx, file_name in enumerate(file_list):
                    processados += 1
                    status_text.text(f"Processando {processados}/{total_files}: {file_name}")
                    progress_bar.progress(processados / total_files)
                    
                    # Extrair arquivo
                    with zip_ref.open(file_name) as file_in_zip:
                        file_bytes = file_in_zip.read()
                        
                        # Criar objeto mock do arquivo
                        class MockUploadedFile:
                            def __init__(self, name, content):
                                self.name = name
                                self._content = content
                            
                            def getvalue(self):
                                return self._content
                            
                            def read(self):
                                return self._content
                        
                        mock_file = MockUploadedFile(
                            name=os.path.basename(file_name),
                            content=file_bytes
                        )
                        
                        # Processar baseado no tipo
                        ext = file_name.split('.')[-1].lower()
                        resultado = {
                            "arquivo": file_name,
                            "tipo": ext,
                            "status": "Desconhecido",
                            "mensagem": ""
                        }
                        
                    # ---------------------------------------------------------------------------- #
                        try:
                            # PDF:
                            if ext in ['pdf']:
                                described_images_hashes = process_pdf(file_path, file_name, described_images_hashes, mock_file=mock_file)
                            else:
                                nao_suportados += 1
                                resultado["status"] = "⚠️ Não suportado"
                                resultado["mensagem"] = f"Tipo de arquivo não suportado: {ext}"
                        
                    # ---------------------------------------------------------------------------- #
                        except Exception as e:
                            com_erro += 1
                            resultado["status"] = "❌ Erro"
                            resultado["mensagem"] = str(e)
                        
                        resultados_detalhados.append(resultado)
        
    # ---------------------------------------------------------------------------- #
        except Exception as e:
            st.error(f"Erro ao processar ZIP: {e}")
        
    # ---------------------------------------------------------------------------- #
        # Limpar progress
        progress_bar.empty()
        status_text.empty()
        
    # ---------------------------------------------------------------------------- #
        # Mostrar resultados
        st.write("---")
        st.subheader("📊 Resumo do Processamento")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Processados", processados)
        with col2:
            st.metric("✅ Sucesso", com_sucesso, delta=None, delta_color="normal")
        with col3:
            st.metric("❌ Erros", com_erro, delta=None, delta_color="inverse")
        with col4:
            st.metric("⚠️ Não Suportados", nao_suportados)
        
        # Tabela de resultados
        if resultados_detalhados:
            st.write("**Detalhes por arquivo:**")
            df_resultados = pd.DataFrame(resultados_detalhados)
            st.dataframe(df_resultados, width='stretch')
            
        #     # Opção de download do relatório
        #     csv = df_resultados.to_csv(index=False).encode('utf-8')
        #     st.download_button(
        #         label="📥 Baixar Relatório CSV",
        #         data=csv,
        #         file_name=f"relatorio_processamento_{uploaded_file.name}.csv",
        #         mime="text/csv"
        #     )
        
    # ---------------------------------------------------------------------------- #
        if com_sucesso > 0:
            st.success(f"🎉 {com_sucesso} documento(s) processado(s) e salvo(s) no banco com sucesso!")
        
        if com_erro > 0:
            st.warning(f"⚠️ {com_erro} arquivo(s) com erro durante o processamento.")
        
        if nao_suportados > 0:
            st.info(f"ℹ️ {nao_suportados} arquivo(s) não suportado(s) ou pulado(s).")

    return described_images_hashes
    
# ============================================================================= #
def main():
    described_images_hashes = {}
    
# ============================================================================= #
    # Streamlit Interface:
    st.title("Multimodal RAG Pipeline - Image Description Agent")
    st.write("This application processes a PDF document, extracts images, and generates descriptions for each image using a language model.")
    
    st.markdown("---")
    
    # --- Tabs para diferentes modos ---
    tab1, tab2 = st.tabs(["📤 Upload de Arquivos", "💬 Chat com IA"])
    
    with tab1:
        # Aba 1: File Upload:
        st.header("1. PDF File Upload")
        uploaded_file = st.file_uploader("Upload a file", type=["pdf", 'zip', 'tar', 'gz', 'tgz', 'rar'])
        
        if uploaded_file is not None:
            st.success(f"✅ Arquivo '{uploaded_file.name}' carregado!")
            
            is_zip = uploaded_file.name.endswith((".zip", ".tar", ".gz", ".tgz", ".rar"))
            is_pdf = uploaded_file.name.endswith(".pdf")

        # ---------------------------------------------------------------------------- #
            if is_pdf:
                # Salva o arquivo temporariamente
                with open(f"{file_path}/{base_file_name}.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # Botão para iniciar o processamento
                if st.button("🚀 Processar PDF", width='stretch'):
                    st.info("Processing the PDF file. This may take a few moments...")
                    described_images_hashes = process_pdf(file_path, base_file_name, described_images_hashes)
                    
        # ---------------------------------------------------------------------------- #
            elif is_zip:
                # Extrai o arquivo zip
                # TODO: Adicionar multi-processing para processar vários documentos ao mesmo tempo
                described_images_hashes = extract_zip(uploaded_file, file_path, described_images_hashes)
            
# ============================================================================= #
    with tab2:
        # Histórico de chat
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        # Exibir mensagens anteriores
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Input do usuário
        if query := st.chat_input("Digite sua pergunta sobre os documentos enviados..."):
            
            # Adicionar mensagem do usuário
            st.session_state.chat_messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            
            # Gerar resposta com agente
            resposta = ""
            with st.chat_message("assistant"):
                with st.spinner("🔍 Consultando banco de dados..."):
                    try:
                        resposta = rag_agent_response(query)
                        # resposta = response.get('output', 'Não consegui gerar uma resposta.')
                    except KeyError as e:
                        resposta = f"Erro de chave: {e}. Verifique se os dados do banco estão no formato correto."
                        st.error("💡 Dica: Pode ser que a estrutura dos dados no banco esteja inconsistente.")
                    except Exception as e:
                        resposta = f"Erro ao processar: {e}"
                        st.error("💡 Dica: Tente reformular sua pergunta de forma mais específica.")
                        
                        # Debug info
                        with st.expander("🔍 Informações de debug"):
                            st.write("**Erro completo:**")
                            st.code(str(e))
                            st.write("**Tipo de erro:**", type(e).__name__)
                
                st.write(resposta)
            
            # Adicionar resposta ao histórico
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": resposta
            })
        
        # Botão para limpar histórico
        if st.button("🗑️ Limpar Conversa"):
            st.session_state.chat_messages = []
            st.session_state.chat_agent_executor.memory.clear()
            st.rerun()
        
# ============================================================================= #
if __name__ == "__main__":
    main()
# from services.rag_service import RAGService