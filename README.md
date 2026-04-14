# Research Corpus Agent

A hybrid system combining **Agentic AI + Retrieval-Augmented Generation** to answer queries and generate structured content grounded in documents.

---

## Features

- Advanced RAG pipeline with:
  - Document retrieval
  - Chunk scoring & evaluation
  - Query rewriting + web fallback
  - Knowledge refinement (decomposition, filtering, recomposition)

- Agent-based content generation:
  - Router → Planner → Workers → Reducer
  - Multi-section structured output

- Evaluation layer:
  - Retrieval score
  - Answer quality scoring
  - Threshold-based validation

- Downloadable outputs (reports & blogs)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd research-corpus-agent

```

### 2. Create virtual environment

```bash
python -m venv venv

venv\Scripts\activate
```

### 3. Install Dependencies

```bash

pip install -r requirements.txt

```

### 4. Add API keys

Create a .env file

``` bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

### 5. Run the app

```bash
streamlit run frontend_research_agent.py
```

## Usage
- Upload PDF documents
- Build knowledge base
- Choose mode:
1. 🔍 Ask Questions (RAG)
2. 📝 Generate Structured Content (Agent)

- Run queries and download results

## Project Structure

<<<<<<< HEAD
agents/        → Agent nodes (router, worker, etc.)
rag/           → RAG pipeline (retrieval, refinement, evaluation)
graphs/        → LangGraph workflow
config/        → LLM + settings
frontend_*.py  → Streamlit UI
=======
1. agents/        → Agent nodes (router, worker, etc.)
2. rag/           → RAG pipeline (retrieval, refinement, evaluation)
3. graphs/        → LangGraph workflow
4. config/        → LLM + settings
5. frontend_*.py  → Streamlit UI
>>>>>>> b750f138d76d82e2431ea130a3255d2d74f1d4b2

## Key innovations

1. Hybrid RAG + Agent system
2. Threshold-based retrieval evaluation
3. Knowledge refinement layer
4. Multi-hop reasoning for complex queries

## Future Improvements
- Add analyst layer (cross-document reasoning)
- Improve evaluation metrics
<<<<<<< HEAD
- Deploy on cloud
=======
- Deploy on cloud
>>>>>>> b750f138d76d82e2431ea130a3255d2d74f1d4b2
