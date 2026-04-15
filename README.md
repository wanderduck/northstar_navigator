# NorthStar Navigator

**[Live Demo](https://navigator.wanderduck.dev)** | **[GGUF Model](https://huggingface.co/wanderduck/northstar-navigator-gguf)** | **[Kaggle Competition](https://www.kaggle.com/competitions/gemma-4-good-hackathon)**

## Introduction
The **NorthStar Navigator** is a privacy-first, AI-powered application developed for the **Kaggle Gemma 4 Good Hackathon**. Navigating government assistance programs can be overwhelming, especially when faced with complex terminology, strict eligibility rules, and dispersed information. This application aims to simplify that process by helping users identify and understand government benefits programs—with a specific focus on Minnesota—using accessible, plain language. 

By leveraging local Large Language Models (LLMs)—specifically targeting the Gemma 4 E4B model—the Navigator ensures that sensitive user situations and personal data remain entirely private and secure on their local machine.

## Overview
The Navigator functions as an intelligent, conversational guide that processes user input and matches it against known government assistance programs (including 15-20 federal programs, 15+ Minnesota state programs, and specific Twin Cities metro county programs). 

To ensure accuracy and approachability, the application relies on an intelligent three-stage pipeline:
1. **Intake:** The system interacts with the user to extract a structured profile, parsing their unique situation and asking clarifying questions if necessary details are missing.
2. **Eligibility:** Using a Retrieval-Augmented Generation (RAG) pipeline, the application cross-references the user's profile against predefined program rules and eligibility criteria to find potential matches.
3. **Response Generation:** The final step translates complex government program details into a plain-language response tailored to the user's reading level and preferred language, complete with rigorous citations.

The application features a clean user interface built with Gradio and relies on Ollama for local model execution, providing a seamless and secure user experience.

## Core Features
*   **Privacy-First Local Execution:** All natural language processing runs locally via Ollama. Sensitive personal, financial, and household details never leave the user's machine.
*   **Intelligent Profile Extraction:** Automatically parses conversational input to build a structured profile, proactively asking clarifying questions to gather required eligibility details.
*   **Targeted Program Matching:** Employs robust hybrid search (vector embeddings combined with keyword search) to evaluate eligibility against federal, state, and county-specific rules.
*   **Adjustable Readability Levels:** Generates responses tailored to specific reading levels (Simple, Standard, Detailed) using the `textstat` library to ensure true "plain language" accessibility.
*   **Mandatory Disclaimers & Safety:** Enforces strict safety rails, ensuring the AI never provides definitive legal or financial determinations, but rather assesses *potential* eligibility and points to official government resources.
*   **Multilingual Support:** Generates responses in English, Spanish, Hmong, and Somali — covering Minnesota's largest LEP communities.

## Getting Started

### Prerequisites
Before running the application, ensure you have the following installed:
*   **Python 3.13+**
*   **`uv`**: The fast Python package manager.
*   **Ollama**: Must be installed and running locally as a service (`systemctl start ollama` on Linux, or via the desktop application).

The application uses a fine-tuned Gemma 4 E4B model (GGUF q4_k_m, 5.3GB). Download and load it:
```bash
hf download wanderduck/northstar-navigator-gguf model-q4_k_m.gguf Modelfile --local-dir output/gguf/gguf
ollama create navigator -f output/gguf/gguf/Modelfile
```

### Installation
Clone the repository and navigate to the project directory. Install the application dependencies using `uv`. Note that navigator dependencies are kept separate from the heavier training stacks to prevent conflicts:

```bash
uv sync --extra navigator
```

### Running the Application
Launch the Gradio user interface by running the main application script. 

*Important: Always use `uv run` to ensure the correct virtual environment is used, and set the protocol buffers environment variable for ChromaDB compatibility:*

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python src/app.py
```
Once the server starts, open your web browser and navigate to `http://0.0.0.0:7860` to access the application.

### Running Tests
To verify your installation and run the project's test suite, you can run `pytest`:

```bash
uv run pytest tests/ -v
```

## Architecture and Data Flow

This section outlines the architecture and data flows of the project across three key domains: RAG & Data Architecture, Core Application Architecture, and Training, Scripts & Deployment.

### 1. RAG & Data Architecture

The knowledge retrieval layer of the NorthStar Navigator relies on a hybrid search architecture designed to handle both semantic queries and exact-match terminology common in government manuals. 

*   **Vector Store & Embeddings**: The project uses **ChromaDB** as its local vector database (`src/navigator/rag/store.py`). Document chunks are embedded using the `sentence-transformers/all-MiniLM-L6-v2` model, which provides a fast, lightweight semantic representation of complex program rules.
*   **Hybrid Search Strategy**: To overcome the limitations of purely semantic search—which often struggles with specific program acronyms (e.g., "MFIP", "CCAP")—the pipeline supplements ChromaDB's vector search with a **BM25 sparse keyword index**. This dual-search approach ensures high recall for both conceptual queries (e.g., "food help") and exact programmatic terms.
*   **Ingestion Pipeline**: The ingestion engine (`scripts/ingest_all.py`) processes multiple data formats. It parses structured JSON files containing core program thresholds (`data/programs/`) and chunks raw scraped HTML/text from external sources, including the Minnesota DHS Combined Manual, county pages, and SAM.gov. As chunks are ingested into ChromaDB, they are tagged with metadata properties like `jurisdiction`, `category`, and `program` to allow for strict pre-filtering during retrieval.

```mermaid
flowchart TD
    subgraph Data ["📁 data/"]
        ChromaDB[("chroma_db/<br>Chroma Vector Store")]
        Programs[/"programs/*.json<br>(Program Rules)"/]
        FPL[/"fpl_tables/fpl_2026.json"/]
        Raw[/"raw/<br>(County/DHS/SAM.gov Data)"/]
    end

    subgraph RAG ["📁 src/navigator/rag/"]
        Ingest["ingest.py<br>(Data Loading & Chunking)"]
        Store["store.py<br>(Semantic Search)"]
        BM25["bm25.py<br>(Keyword Search)"]
        Embeddings["embeddings.py<br>(all-MiniLM-L6-v2)"]
        Retrieval["retrieval.py<br>(HybridRetriever)"]
    end

    subgraph Tools ["📁 src/navigator/tools/"]
        BenefitsSearch["benefits_search.py<br>(BenefitsSearchTool)"]
        CountyPrograms["county_programs.py<br>(CountyProgramsTool)"]
    end

    Ingest -->|Reads| Programs
    Ingest -->|Reads| Raw
    Ingest -->|Writes & Adds Metadata| ChromaDB
    Ingest -->|Builds Index| BM25
    
    Store -->|Reads/Writes| ChromaDB
    Store -->|Uses| Embeddings
    
    Retrieval -->|Queries| Store
    Retrieval -->|Queries| BM25
    Retrieval -->|Merges Scores & Filters| Retrieval
    
    BenefitsSearch -->|Uses| Retrieval
    CountyPrograms -->|Uses| Retrieval
    BenefitsSearch -->|Reads| FPL
```

### 2. Core Application Architecture

The application orchestrates a three-stage pipeline to bridge the gap between unstructured user queries and rigorous government criteria, all wrapped in a responsive user interface.

*   **User Interface**: **Gradio** powers the frontend (`src/app.py`), providing an accessible, chat-like interface. Gradio's sidebar components are utilized to let users dynamically control the application's configuration, including selecting a target reading level ("Simple", "Standard", "Detailed") and language translation (English, Spanish, Hmong, Somali, Karen).
*   **Stage 1: Intake**: When a user submits a query, the `IntakeProcessor` (`src/navigator/intake.py`) uses a local LLM via **Ollama** to parse the unstructured text into a strongly typed `UserProfile` Pydantic model. The LLM extracts demographic, financial, and geographic data, and can optionally flag missing information to ask clarifying questions.
*   **Stage 2: Eligibility Engine**: The `EligibilityEngine` (`src/navigator/eligibility.py`) serves as the core logic hub. It intentionally offloads deterministic calculations from the LLM to Python-based tools. It calculates Federal Poverty Level (FPL) thresholds, age restrictions, and household size checks using hard-coded functions. For nuanced requirements, it triggers the RAG pipeline to search ChromaDB via Tools. The engine compiles these checks into a structured `BenefitsResponse` object.
*   **Stage 3: Response Generation**: Finally, the `ResponseGenerator` (`src/navigator/response.py`) takes the structured `BenefitsResponse` and feeds it back into the local LLM via **Ollama**. Bound by a strict system prompt tailored to the Gradio-selected reading level and language, the model generates an empathetic, plain-language summary of potential eligibility. The system automatically appends a mandatory legal disclaimer to ensure the application only advises on *potential* eligibility rather than guaranteeing benefits.

```mermaid
flowchart TD
    %% File System Structure using Subgraphs
    subgraph src ["📁 src/"]
        App["app.py<br/>(Gradio UI & Orchestrator)"]
        
        subgraph navigator ["📁 navigator/"]
            subgraph pipeline ["🔄 Pipeline Stages"]
                Intake["intake.py<br/>(Stage 1: IntakeProcessor)"]
                Eligibility["eligibility.py<br/>(Stage 2: EligibilityEngine)"]
                ResponseGen["response.py<br/>(Stage 3: ResponseGenerator)"]
            end
            
            subgraph shared ["⚙️ Shared Infrastructure"]
                Models["models.py<br/>(UserProfile, BenefitsResponse)"]
                Ollama["ollama_client.py<br/>(Local LLM Wrapper)"]
                Prompts["prompts.py<br/>(System Prompts & Disclaimers)"]
                Config["config.py<br/>(Settings & Constants)"]
                Readability["readability.py<br/>(Textstat Utils)"]
            end
            
            Tools["📁 tools/<br/>(BenefitsSearchTool, CountyProgramsTool)"]
        end
    end

    %% Pipeline Orchestration / Data Flow
    App == "1. Raw user message" ==> Intake
    Intake == "2. Extracted UserProfile" ==> App
    App == "3. UserProfile" ==> Eligibility
    Eligibility == "4. BenefitsResponse<br/>(Determined Programs)" ==> App
    App == "5. UserProfile + BenefitsResponse" ==> ResponseGen
    ResponseGen == "6. Plain-language<br/>Markdown Response" ==> App

    %% Component Dependencies
    App -. "Configures UI State" .-> Models

    Intake -. "Parses via" .-> Ollama
    Intake -. "Uses" .-> Prompts

    Eligibility -. "Evaluates Rules & Uses" .-> Tools
    Eligibility -. "Builds" .-> Models

    ResponseGen -. "Generates via" .-> Ollama
    ResponseGen -. "Enforces Tone & Disclaimers" .-> Prompts
    ResponseGen -. "Adjusts Level" .-> Readability

    style App fill:#f9f,stroke:#333,stroke-width:2px
    style Intake fill:#bbf,stroke:#333
    style Eligibility fill:#bbf,stroke:#333
    style ResponseGen fill:#bbf,stroke:#333
```

### 3. Training, Scripts, and Deployment

To ensure the model accurately translates dense bureaucratic language into accessible prose without hallucinating eligibility criteria, the project relies on a localized fine-tuning and deployment toolchain.

*   **Parameter-Efficient Fine-Tuning (PEFT)**: The project uses **Unsloth** in conjunction with the `trl` `SFTTrainer` (`training/train_unsloth.py`) to fine-tune the Gemma 4 E4B model. By utilizing QLoRA (Quantized Low-Rank Adaptation), Unsloth heavily optimizes GPU memory consumption, allowing the model to be fine-tuned locally on an RTX 2080 Ti or in the cloud via Kaggle's T4 instances and Modal A100 deployments. 
*   **Training Data**: The models are trained on specialized JSONL datasets (`data/training/`) that map confusing DHS manual excerpts to their corresponding plain-language translations across multiple languages and edge-case scenarios. 
*   **GGUF Export & Inference Deployment**: Following fine-tuning, Unsloth's export utilities (`training/export_gguf.py`) package the updated model weights into the quantized GGUF format. 
*   **Local Privacy via Ollama**: The resulting GGUF artifact is served locally using **Ollama** (`src/navigator/ollama_client.py`). Because Ollama runs entirely on the host machine, all highly sensitive user profile data (e.g., household income, disability status, immigration status) is processed locally. This architecture guarantees that no personally identifiable information (PII) is ever transmitted to a third-party cloud provider, adhering to strict data privacy and safety standards.

```mermaid
graph TD
    classDef script fill:#f9d0c4,stroke:#333,stroke-width:1px;
    classDef data fill:#d4e6f1,stroke:#333,stroke-width:1px;
    classDef deploy fill:#d5f5e3,stroke:#333,stroke-width:1px;
    classDef train fill:#fcf3cf,stroke:#333,stroke-width:1px;
    classDef notebook fill:#ebdef0,stroke:#333,stroke-width:1px;

    subgraph Data Layer
        D_RAW["data/raw/ <br/>(County pages, DHS, SAM.gov)"]:::data
        D_PROG["data/programs/*.json <br/>(Program rules)"]:::data
        D_DB["data/chroma_db/ <br/>(RAG Vector Store)"]:::data
        D_TRN_BATCH["data/training/batch_*.jsonl"]:::data
        D_TRN_FINAL["data/training/final.jsonl"]:::data
    end

    subgraph Scripts & Ingestion
        S_SCRAPE["scripts/scrape_*.py <br/>scripts/download_sam_gov.py"]:::script
        S_BATCH["scripts/batch_extract_dhs.py <br/>scripts/extract_dhs_page.py"]:::script
        S_INGEST["scripts/ingest_all.py"]:::script
        
        S_SCRAPE -->|Scrape web content| D_RAW
        S_BATCH -->|Format JSON to txt| D_RAW
        D_PROG --> S_INGEST
        D_RAW --> S_INGEST
        S_INGEST -->|Create embeddings| D_DB
    end

    subgraph Model Training
        T_PREP["training/prepare_dataset.py"]:::train
        T_TRAIN["training/train_unsloth.py<br/>(Unsloth QLoRA)"]:::train
        T_EXPGGUF["training/export_gguf.py"]:::train
        T_KAGGLE["training/kaggle_finetune.ipynb"]:::notebook
        
        D_TRN_BATCH --> T_PREP
        T_PREP -->|Combine, validate, dedup| D_TRN_FINAL
        D_TRN_FINAL --> T_TRAIN
        D_TRN_FINAL --> T_KAGGLE
        T_TRAIN -->|Fine-tune model| T_EXPGGUF
        T_EXPGGUF -->|Modelfile & .gguf| M_OUT["Ollama / GGUF Model Artifacts"]:::data
    end

    subgraph Deployment
        D_MODAL_FT["deploy/modal_finetune.py<br/>(Unsloth A100)"]:::deploy
        D_MODAL_FT_PLAIN["deploy/modal_finetune_plain.py<br/>(TRL Fallback)"]:::deploy
        D_MODAL_APP["deploy/modal_app.py<br/>(Modal Gradio)"]:::deploy
        D_FLASH["deploy/flash_navigator.py<br/>(RunPod Flash)"]:::deploy
        
        D_TRN_FINAL --> D_MODAL_FT
        D_TRN_FINAL --> D_MODAL_FT_PLAIN
        D_MODAL_FT -->|Cloud Fine-tuning| M_OUT
        D_MODAL_FT_PLAIN -->|Cloud Fine-tuning| M_OUT
        
        D_DB --> D_MODAL_APP
        D_PROG --> D_MODAL_APP
        M_OUT -.->|Provides Model| D_MODAL_APP
        M_OUT -.->|Provides Model| D_FLASH
    end
```