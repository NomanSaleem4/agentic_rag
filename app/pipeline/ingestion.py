import os, time, json, re
from pathlib import Path
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from loguru import logger

import psycopg2
from docling.document_converter import DocumentConverter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext



load_dotenv()
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "")
PG_USER = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "llama_index_docs")

PDF_INPUT_FOLDER = os.getenv("PDF_INPUT_FOLDER", "data/docs_to_ingest")
MARKDOWN_FOLDER = os.getenv("MARKDOWN_FOLDER", "data/processed_markdown_data")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
GENERATE_CONTEXT = os.getenv("GENERATE_CONTEXT", "true").lower() in ("1","true","yes")
CONTEXT_MAX_RETRIES = int(os.getenv("CONTEXT_MAX_RETRIES", "2"))
CONTEXT_WHOLE_DOC_MAX = int(os.getenv("CONTEXT_WHOLE_DOC_MAX", "2000"))
CONTEXT_CHUNK_MAX = int(os.getenv("CONTEXT_CHUNK_MAX", "500"))

CLEAR_TABLE_BEFORE_INGEST = os.getenv("CLEAR_TABLE_BEFORE_INGEST", "false").lower() in ("1","true","yes")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

HNSW_KWARGS = {
    "hnsw_m": int(os.getenv("HNSW_M", "16")),
    "hnsw_ef_construction": int(os.getenv("HNSW_EF_CONSTRUCTION", "64")),
    "hnsw_ef_search": int(os.getenv("HNSW_EF_SEARCH", "40")),
}



def convert_pdfs_to_markdown(input_folder: str = PDF_INPUT_FOLDER, output_folder: str = MARKDOWN_FOLDER) -> List[str]:
    os.makedirs(output_folder, exist_ok=True)
    out = Path(output_folder)
    pdfs = list(Path(input_folder).glob("*.pdf"))
    if not pdfs:
        logger.info("No PDFs to convert.")
        return []
    conv = DocumentConverter()
    saved=[]
    for p in pdfs:
        try:
            r = conv.convert(p)
            md = r.document.export_to_markdown()
            path = out / (p.stem + ".md")
            path.write_text(md, encoding="utf-8")
            saved.append(str(path))
            logger.info(f"Converted {p.name}")
        except Exception as e:
            logger.warning(f"Convert failed {p.name}: {e}")
    return saved

def chunk_markdown_folder(markdown_folder: str = MARKDOWN_FOLDER) -> List[Dict[str,Any]]:
    p = Path(markdown_folder)
    if not p.exists():
        raise FileNotFoundError(f"{markdown_folder} not found")
    docs = SimpleDirectoryReader(markdown_folder, recursive=False).load_data()
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(nodes)
    chunks=[]
    for i,n in enumerate(nodes, start=1):
        meta = dict(n.metadata or {})
        # ensure JSON-serializable
        try: json.dumps(meta)
        except: meta = {k:str(v) for k,v in meta.items()}
        chunks.append({"id": i, "text": n.text, "metadata": meta})
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def generate_context_for_chunk(whole_document: str, chunk_content: str, max_retries: int = CONTEXT_MAX_RETRIES) -> str:
    whole = (whole_document[:CONTEXT_WHOLE_DOC_MAX] + "...") if whole_document and len(whole_document) > CONTEXT_WHOLE_DOC_MAX else (whole_document)
    chunk = (chunk_content[:CONTEXT_CHUNK_MAX] + "...") if chunk_content and len(chunk_content) > CONTEXT_CHUNK_MAX else (chunk_content)
    prompt = f"Document: {whole}\nChunk: {chunk}\nContext:"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.3, "num_predict": 60, "num_ctx":3000, "stop":["\n","."]}}
    headers={"Content-Type":"application/json"}
    for attempt in range(1, max_retries+1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=25)
            if r.status_code == 200:
                j = r.json()
                ctx = j.get("response","") or j.get("text","") or ""
                ctx = ctx.strip()
                ctx = re.sub(r'^["\']|["\']$', '', ctx).replace("\n"," ").strip()
                if ctx and ctx.lower() != "context generation failed":
                    return ctx
        except Exception as e:
            logger.warning(f"Ollama attempt {attempt} failed: {e}")
            time.sleep(1)
    return ""


def create_pgvector_store_and_index(chunks: List[Dict[str,Any]]):
    logger.info("Creating PGVectorStore (perform_setup=True)...")
    pg_store = PGVectorStore.from_params(
        database=PG_DATABASE, host=PG_HOST, password=PG_PASSWORD, port=PG_PORT, user=PG_USER,
        table_name=PG_TABLE_NAME, embed_dim=EMBED_DIM, hybrid_search=True, text_search_config="english",
        perform_setup=True, hnsw_kwargs=HNSW_KWARGS
    )
    if CLEAR_TABLE_BEFORE_INGEST:
        try:
            pg_store.clear()
            logger.info("Cleared existing vector store table.")
        except Exception as e:
            logger.warning(f"Could not clear vector store: {e}")

    hf_embed = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    from llama_index.core import Settings
    Settings.embed_model = hf_embed

    docs = []
    for c in chunks:
        txt = c.get("text", "")
        meta = dict(c.get("metadata") or {})
        file_path = meta.get("file_path") or meta.get("source") or ""
        whole_doc = ""
        if file_path and Path(file_path).exists():
            try:
                whole_doc = Path(file_path).read_text(encoding="utf-8")
            except:
                whole_doc = ""
        contextual_text = generate_context_for_chunk(whole_doc, txt) if GENERATE_CONTEXT else ""
        if not contextual_text:
            contextual_text = f"Document section"
        combined = f"{txt}\n\nContext: {contextual_text}"
        meta["original_text"] = txt
        meta["contextual_text"] = contextual_text
        docs.append(Document(text=combined, metadata=meta))

    storage_context = StorageContext.from_defaults(vector_store=pg_store)
    logger.info(f"Indexing {len(docs)} documents (LlamaIndex will compute embeddings).")
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    logger.success("Indexing complete.")
    return index


def run_ingest():
    md = Path(MARKDOWN_FOLDER)
    if not md.exists():
        logger.info("Converting PDFs to markdown...")
        converted = convert_pdfs_to_markdown(PDF_INPUT_FOLDER, MARKDOWN_FOLDER)
        if not converted:
            logger.warning("No markdown produced; aborting."); return
    chunks = chunk_markdown_folder(MARKDOWN_FOLDER)
    if not chunks:
        logger.warning("No chunks; aborting."); return
    return create_pgvector_store_and_index(chunks)

if __name__ == "__main__":
    logger.info("Starting ingestion...")
    run_ingest()
   
