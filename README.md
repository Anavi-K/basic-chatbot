# AI Chatbot — RAG-Powered LLM Application

A GenAI chatbot built with **Python, Gemini, LangChain, and ChromaDB**. The application combines RAG-based knowledge retrieval with tool calling for calculations, structured CSV analysis, and other utility tasks.

## Live Demo

* **Frontend:** https://chatbot-fe-w78w.onrender.com/
* **Backend:** https://chatbot-xven.onrender.com/

## Features

* **RAG Knowledge Base** — Retrieves relevant information from a local ChromaDB vector database before generating answers.
* **LLM Tool Calling** — Uses LangChain tools for calculations, structured data analysis, text utilities, web search, and knowledge-base retrieval.
* **Hallucination Guardrails** — Rejects questions when retrieved context does not meet the required relevance threshold.
* **Structured Data Analysis** — Uses Pandas to analyse sales data and perform operations such as sum, average, maximum, minimum, grouped aggregation, and top-N queries.
* **Gemini Integration** — Uses Google's Gemini API for natural-language understanding and response generation.

## GenAI / RAG Architecture

```text
User Query
    ↓
LangChain Agent
    ↓
Tool Selection
    ├── Calculator
    ├── Sales Data Analysis
    ├── Text Utilities
    ├── Web Search
    └── Knowledge Base Search
              ↓
        ChromaDB Retrieval
              ↓
        Relevance Check
          ↙         ↘
      Relevant    Insufficient
         ↓             ↓
      Gemini       Guardrail
         ↓             ↓
    Grounded       "I don't have
     Answer         enough information..."
```

## Knowledge Base

The RAG system currently contains **4 knowledge-base documents**:

* `company_overview.txt`
* `return_shipping_policy.txt`
* `product_catalog_info.txt`
* `faq.txt`

These documents are split into **21 text chunks** using LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 400 and 50-character overlap.

The chunks are embedded and persisted locally using **ChromaDB** for semantic retrieval.

## Hallucination Guardrails

The application calculates retrieval relevance before generating a knowledge-based response.

A relevance threshold of **0.40** is used.

If retrieved context does not meet the threshold, the chatbot does not generate an unsupported answer and instead returns:

> "I don't have enough information in my knowledge base to answer that."

## Evaluation

The project includes a reproducible evaluation script in `eval.py`.

### Results

| Metric                           |           Result |
| -------------------------------- | ---------------: |
| Knowledge Retrieval Precision    | **100% (10/10)** |
| Average Context Relevance        |       **0.6028** |
| Hallucination Guardrail Accuracy |   **100% (5/5)** |
| Multi-Tool Preservation Rate     |         **100%** |

The evaluation includes:

* **10** grounded knowledge-base queries
* **5** out-of-domain queries
* Verification that existing tools continue to function alongside the RAG system

## Tech Stack

**Language**

* Python

**GenAI**

* Google Gemini
* LangChain
* RAG
* ChromaDB
* Embeddings
* Tool Calling

**Data**

* Pandas
* CSV

**Frontend**

* React.js

**Backend**

* Flask
* REST API

## Project Structure

```text
basic-chatbot/
├── backend/
│   ├── data/
│   │   ├── knowledge/
│   │   ├── chroma_db/
│   │   └── sales.csv
│   ├── rag_manager.py
│   ├── eval.py
│   └── ...
├── frontend/
│   └── ...
└── README.md
```

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Run the backend:

```bash
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Evaluation

Run the evaluation suite from the backend:

```bash
python eval.py
```

This reproduces the retrieval, guardrail, and tool-preservation metrics reported above.
