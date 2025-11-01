# Agentic RAG Application

An agentic RAG (Retrieval-Augmented Generation) system with FastAPI, OpenWebUI, and PostgreSQL/Pgvector storage.

## Architecture

- **FastAPI**: OpenAI compatible RAG API server with CrewAI agent
- **OpenWebUI**: Web interface for chat interactions
- **PostgreSQL + pgvector**: Vector database for document embeddings and text search
- **Phoenix**: Observability and tracing


## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL + pgvector installed and running
- Ollama up and running
- Arize Phoenix installed and running

### Setup Ubuntu

1. **Clone and navigate to the project:**
   ```bash
   cd /your_local_machine_pre_path/agentic_rag
   export PYTHONPATH=/your_local_machine_pre_path/agentic_rag
   ```

2. **Create environment file:**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key

    ```

3. **Install Postgress and PGVector extension**
   **Install PostgreSQL**
   ```bash
   sudo apt update
   sudo apt install postgresql postgresql-contrib -y
   sudo systemctl start postgresql
   ```

   **Install pgvector**
   ```bash
   sudo apt install build-essential git postgresql-server-dev-all -y
   cd /tmp
   git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

   **Setup Database**
   ```bash
   sudo -u postgres psql
   ```
   ```sql
   CREATE USER your_user WITH PASSWORD 'your_password';
   CREATE DATABASE your_db;
   GRANT ALL PRIVILEGES ON DATABASE your_db TO your_user;
   \c your_db
   CREATE EXTENSION vector;
   \q
   ```
   **Configuration**
   Update `.env` or config file:
   ```
   PG_HOST=localhost
   PG_PORT=5432
   PG_DATABASE=your_db
   PG_USER=your_user
   PG_PASSWORD=your_password
   ```

   **Check if running**
   ```
   sudo systemctl status postgresql
   ```
   **Stop PostgreSQL**
   ```
   sudo systemctl stop postgresql

   ```
   **Start PostgreSQL**
   ```
   sudo systemctl start postgresql
   ```

4. **Run Ollama locally**
   ```
   # Quick install (recommended)
   curl -fsSL https://ollama.com/install.sh | sh

   # Or manual install
   curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
   chmod +x ollama
   sudo mv ollama /usr/local/bin/
   ```

5. **Run Open WebUI locally**
   ```
   sudo docker run    --name openwebui   --network host   -e OLLAMA_BASE_URL=http://localhost
:11434   -e OPENAI_API_BASE_URL=http://localhost:8000/v1   -e OPENAI_API_KEY=your-api-key-here   -e WEBUI_SECRET_KEY=your-secret-key-here   -e ENABLE
_FOLLOW_UP_GENERATION=False   ghcr.io/open-webui/open-webui:slim
   ```


6. **Run Arize Phoenix locally**
   ```
   <!-- docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest -->

   docker run -d -p 6006:6006 -p 4317:4317 -v $(pwd)/phoenix-data:/data arizephoenix/phoenix:latest
   ```

7. **Run Documents Ingestion Pipeline**
   ```
   python -m app.pipeline.ingestion
   ```

8. **Start RAG and Open WebUI services:**
   ```
   docker-compose up -d
   ```

9. **Access the applications:**
   - **OpenWebUI**: http://localhost:8080


10. **Run RAGAS Evaluation**
   python -m app.evaluation.rag_evaluation



**Ingestion Process:**
1. **PDF Conversion**: Converts PDFs to Markdown format
2. **Chunking**: Splits Markdown into text chunks
3. **Context Generation**: Uses Ollama Gemma2:2b to generate contextual descriptions
4. **Embedding**: Creates vector embeddings for combined text (chunk + context)
5. **Storage**: Stores everything in PostgreSQL with pgvector


### API Endpoints

- `POST /v1/chat/completions`: Chat completion (OpenAI compatible)
- `GET /health`: Health check
- `GET /v1/models`: List available models


### Service Management

**Stop all services:**
```bash
# Stop Docker services
docker-compose down

# Stop system services
sudo systemctl stop postgresql
sudo systemctl stop ollama
```

**Stop individual services:**
```bash
# Stop FastAPI server (if running manually)
lsof -t -i:8000 | xargs kill -9

# Stop OpenWebUI (if running manually)
pkill -f "open_webui"
```

