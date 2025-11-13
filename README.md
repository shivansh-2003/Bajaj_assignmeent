# RAG Document Retrieval System

A minimal RAG (Retrieval-Augmented Generation) system for document search and question answering using Chroma vector database and Ollama LLM.

## Features

- 📄 **Document Ingestion**: Load and process PDF/DOCX files into Chroma vector database
- 🔍 **Document Search**: Semantic search across ingested documents
- 💬 **Question Answering**: Answer questions using retrieved context with optional LLM generation
- 🎨 **Streamlit UI**: Simple web interface for querying documents

## Project Structure

```
RBAC/
├── ingest.py      # Document ingestion pipeline
├── retrieve.py     # RAG retrieval and Q&A pipeline
├── app.py         # Streamlit web interface
└── chroma_db/     # Chroma vector database storage
```

## Requirements

### Python Packages

```bash
pip install streamlit langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama langchain-text-splitters chromadb sentence-transformers httpx
```

### External Services

- **Ollama** (optional, for LLM generation):
  ```bash
  # Install Ollama from https://ollama.ai
  ollama serve
  ollama pull llama3  # or mistral, gpt-oss:20b, etc.
  ```

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Start Ollama server:
   ```bash
   ollama serve
   ```

## Usage

### 1. Ingest Documents (`ingest.py`)

Load documents into the vector database:

```python
from ingest import RAGIngestionPipeline

pipeline = RAGIngestionPipeline()
pipeline.ingest_file("document.pdf", collection_name="my_collection")
```

**Command Line:**
```bash
python ingest.py
# Enter file path when prompted
```

**Supported Formats:**
- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)

### 2. Retrieve & Query (`retrieve.py`)

Search documents or answer questions:

```python
from retrieve import RAGRetrievalPipeline

pipeline = RAGRetrievalPipeline()

# Search documents
results = pipeline.search("query text", collection_name="my_collection", top_k=4)

# Answer question with LLM
answer = pipeline.answer_question(
    "Who is Renly Baratheon?",
    collection_name="my_collection",
    use_llm=True,
    top_k=4
)

# Answer question without LLM (context only)
docs = pipeline.answer_question(
    "Who is Renly Baratheon?",
    collection_name="my_collection",
    use_llm=False,
    top_k=4
)
```

**Command Line:**
```bash
python retrieve.py
# Enter query and collection name when prompted
```

### 3. Streamlit Web Interface (`app.py`)

Launch the web interface:

```bash
streamlit run app.py
```

**Features:**
- Collection selection
- LLM toggle (enable/disable)
- Top-K document count slider
- Query input and search

## Configuration

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`

Edit in `retrieve.py` and `ingest.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Ollama Model

Default: `gpt-oss:20b`

Edit in `retrieve.py`:
```python
OLLAMA_MODEL = "gpt-oss:20b"  # Change to llama3, mistral, etc.
```

### Chroma Database Directory

Default: `./chroma_db`

Edit in both files:
```python
CHROMA_DIR = "./chroma_db"
```

## How It Works

1. **Ingestion** (`ingest.py`):
   - Loads documents (PDF/DOCX)
   - Splits into chunks (1000 chars, 200 overlap)
   - Generates embeddings using HuggingFace
   - Stores in Chroma vector database

2. **Retrieval** (`retrieve.py`):
   - Takes user query
   - Generates query embedding
   - Searches similar chunks in Chroma
   - Optionally generates answer using Ollama LLM

3. **Web Interface** (`app.py`):
   - Provides UI for querying
   - Displays results
   - Configures collection and parameters

## Collections/Namespaces

Documents are organized into collections (similar to namespaces). Use different collection names to separate document sets:

```python
# Ingest into different collections
pipeline.ingest_file("hr_docs.pdf", collection_name="hr")
pipeline.ingest_file("finance_docs.pdf", collection_name="finance")

# Query specific collection
results = pipeline.search("query", collection_name="hr")
```

## Error Handling

The system gracefully handles:
- **Ollama not running**: Falls back to context-only retrieval
- **Model not found**: Shows helpful error message with suggestions
- **Connection errors**: Provides clear error messages

## Example Workflow

```bash
# 1. Ingest a document
python ingest.py
# Enter: gamesofthrones.pdf
# Enter collection: got

# 2. Query via command line
python retrieve.py
# Enter query: Who is Renly Baratheon?
# Enter collection: got

# 3. Or use Streamlit UI
streamlit run app.py
# Use the web interface to query
```

## Troubleshooting

### "Cannot connect to Ollama server"
- Make sure Ollama is running: `ollama serve`
- Check if Ollama is installed: `ollama --version`

### "Model not found"
- List available models: `ollama list`
- Pull a model: `ollama pull llama3`

### "Collection not found"
- Make sure documents are ingested into the collection
- Check collection name spelling
- Default collection is `"default"`

### Import errors
- Install missing packages: `pip install <package-name>`
- For `langchain-chroma`: `pip install langchain-chroma`

## License

MIT License

## Author

RBAC Project

