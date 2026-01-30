# Simple RAG System (LangChain + Chroma + OpenAI)

This is a simple pet project created to understand the basics of **Retrieval-Augmented Generation (RAG)** systems.

The project demonstrates how to:
- load data from the web,
- split it into chunks,
- store embeddings in a vector database (Chroma),
- retrieve relevant context for a user query,
- and generate an answer using an LLM with retrieved context.

---

## Tech Stack

- Python 3.10+
- LangChain
- OpenAI API
- Chroma (vector database)
- BeautifulSoup (HTML parsing)
- python-dotenv

---

## Project Structure
.
├── scripts/
│ ├── api.py # Loads OpenAI API key from environment variables
│ ├── rag_data.py # Downloads data, splits it, creates and persists Chroma DB
│ ├── query_rag.py # Runs a sample RAG query against the stored data
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md


---

## How It Works

1. **Data ingestion**
   - `rag_data.py` downloads a blog post about prompt engineering from the web.
   - HTML is filtered using BeautifulSoup.
   - Text is split into overlapping chunks.
   - Each chunk is embedded using OpenAI embeddings.
   - Embeddings are stored locally in a Chroma vector database.

2. **Querying (RAG)**
   - `query_rag.py` loads the persisted Chroma database.
   - Retrieves the top-k most relevant chunks for a user question.
   - Injects retrieved context into a prompt.
   - Sends the augmented prompt to an OpenAI chat model.
   - Prints the final answer.

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a .env file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
You can use .env.example as a template.


## Run the Project
### 1. Create vector database

```bash
python rag_data.py
```

This will:

download the source document,

split it into chunks,

create and persist a Chroma database in ./chroma_db.


### 2. Run a RAG query

```bash
python query_rag.py
```
The script runs a sample question and prints the LLM answer generated using retrieved context.

## Notes

This project is intended for learning purposes.

API keys are stored securely using environment variables.

The vector database is persisted locally for reuse between runs.


## Author

Denys Laptiev

Pet project created for learning RAG systems and LangChain fundamentals.