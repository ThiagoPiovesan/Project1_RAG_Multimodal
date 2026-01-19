# 🧠 Multimodal RAG Agent: Document Analysis with Vision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green)
![Unstructured](https://img.shields.io/badge/Unstructured.io-Parsing-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-yellow)

## 📋 Sobre o Projeto

Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation) Multimodal** capaz de ingerir, processar e "raciocinar" sobre documentos complexos (PDFs) que contêm não apenas texto, mas também **tabelas estruturadas e imagens/gráficos**.

Diferente de sistemas RAG tradicionais que ignoram informações visuais ou quebram tabelas, este pipeline utiliza modelos de **Visão Computacional** e estratégias de **Parsing Semântico** para garantir que nenhum contexto seja perdido.

### 🎯 Principais Diferenciais

* **Análise Visual:** Extração e descrição automática de imagens e gráficos usando LLMs de Visão (VQA).
* **Preservação de Layout:** Uso de estratégia `hi_res` para extrair tabelas mantendo sua estrutura HTML, permitindo que a IA responda perguntas sobre dados tabulares com alta precisão.
* **Arquitetura Agêntica:** Implementação de um Agente ReAct (Reason+Act) que decide quando consultar a base de conhecimento vetorial.
* **Busca Semântica:** Indexação vetorial híbrida (texto + descrições visuais) utilizando FAISS.

---

## 🏗️ Arquitetura da Solução

O pipeline de dados segue o fluxo abaixo:

1. **Ingestão:** Upload de PDFs complexos.
2. **Parsing Multimodal (Unstructured.io):**
    * Separação de elementos: Texto Narrativo, Tabelas e Imagens.
    * **Chunking Semântico:** Uso de `chunk_by_title` para preservar contexto de seções.
3. **Enriquecimento (Vision Pipeline):**
    * Imagens são convertidas para Base64.
    * LLM Vision gera descrições detalhadas (captions) dos elementos visuais.
    * Tabelas são convertidas para HTML limpo.
4. **Indexação:**
    * Geração de Embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
    * Armazenamento em banco vetorial local (**FAISS**).
5. **Recuperação e Resposta:**
    * Agente LangChain recebe a query do usuário.
    * Ferramenta de busca recupera top-k contextos relevantes.
    * LLM sintetiza a resposta final citando fontes.

---

## 🛠️ Tech Stack

* **Linguagem:** Python
* **Orquestração:** LangChain / LangGraph
* **Parsing & ETL:** Unstructured.io (Detectron2/YOLOX under the hood)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **LLM & Vision:** [Inserir Modelo, ex: GPT-4o / Gemini 1.5 Flash]
* **Interface:** Streamlit (opcional)

---

## 📂 Estrutura do Projeto

```bash
multimodal-rag/
├── data/                   # Diretório para PDFs de entrada
│   ├── temp_images.py      # Diretório de imagens temporárias
├── agents/
│   ├── image_descriptor.py # Pipeline do Vision Description
│   └── rag_agent.py        # Implementação do Agente RAG
├── rag/
│   ├── faiss_rag_index/    # Persistência do Banco Vetorial
│   ├── vector_store.py     # Lógica de Embeddings e FAISS
├── main_interface.py       # Interface Streamlit e Lógica do Agente
├── requirements.txt        # Dependências
├── pyproject.toml          # Dependências
└── README.md
```

---

## 🚀 Como Executar

### 🌐Pré-requisitos

* Python 3.10+
* Chave de API configurada (OpenAI/Google/Anthropic) no arquivo ```.env```

### 💻Instalação

1. Clone o repositório:

    ```Bash
    git clone [https://github.com/seu-usuario/multimodal-rag-agent.git](https://github.com/seu-usuario/multimodal-rag-agent.git)
    cd multimodal-rag-agent
    ```

2. Instale as dependências (incluindo bibliotecas de OCR/Visão):

    ```Bash
    pip install -r requirements.txt
    # Instalação adicional para o Unstructured (sistema operacional)
    # sudo apt-get install poppler-utils tesseract-ocr
    ```

3. Inicie a aplicação:

    ```Bash
    streamlit run main_interface.py
    # Ou
    uv run streamlit run main_interface.py
    ```

---

### 🔮 Próximos Passos & Melhorias

* **Re-ranking:** Implementar um passo de Cross-Encoder (ex: BGE-Reranker) após a busca no FAISS para refinar a relevância dos documentos entregues à LLM.

* **Avaliação (Ragas):** Criar um pipeline de testes automatizados para medir a precisão (faithfulness) e relevância das respostas.

* **Modelos Locais:** Substituir a API de Visão por modelos open-source rodando localmente (ex: LLaVA ou Florence-2) para privacidade total dos dados.

* **Deploy:** Containerização da aplicação com Docker.

---

#### 🤝 Contato

Thiago Piovesan Engenheiro de IA | Especialista em Visão Computacional [LinkedIn](https://www.linkedin.com/in/thiago-piovesan/) | [Portfólio](https://github.com/ThiagoPiovesan)
