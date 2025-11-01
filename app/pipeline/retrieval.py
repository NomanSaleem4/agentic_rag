import os
from typing import List, Dict, Any, Union
import psycopg2
import numpy as np
from dotenv import load_dotenv
import requests
from sentence_transformers import CrossEncoder
from loguru import logger
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import make_url


# import shutil
# from crewai.utilities.paths import db_storage_path

# # Get storage path
# storage_path = db_storage_path()

# # Delete existing memory storage
# if os.path.exists(storage_path):
#     shutil.rmtree(storage_path)
#     print(f"\n Cleared memory storage at: {storage_path}")

# Phoenix configuration
PHOENIX_API_URL = os.getenv("PHOENIX_API_URL", "http://localhost:6006")

# LLM configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
logger.info(f"USE_OLLAMA: {USE_OLLAMA}")

# Load environment variables from .env file
load_dotenv()

# Database connection details
pg_host = os.getenv("PG_HOST", "localhost")
pg_port = os.getenv("PG_PORT", 5432)
pg_database = os.getenv("PG_DATABASE")
pg_user = os.getenv("PG_USER")
pg_password = os.getenv("PG_PASSWORD")



# Generate query embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# # Find the exact path
# cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
# model_path = os.path.join(cache_dir, "models--sentence-transformers--all-MiniLM-L6-v2")

# # Check subdirectories for the snapshot
# if os.path.exists(model_path):
#     snapshots = os.path.join(model_path, "snapshots")
#     if os.path.exists(snapshots):
#         snapshot_dirs = os.listdir(snapshots)
#         if snapshot_dirs:
#             full_model_path = os.path.join(snapshots, snapshot_dirs[0])
            
#             embed_model = HuggingFaceEmbedding(
#                 model_name=full_model_path
#             )

# Connection
connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
url = make_url(connection_string)


from llama_index.llms.ollama import Ollama

llama_index_llm_ = Ollama(
    model= "qwen2.5:7b", #"mistral:7b-instruct-q2_K",
    base_url="http://localhost:11434",
    temperature=0.1,
)

HNSW_KWARGS = {
    "hnsw_m": int(os.getenv("HNSW_M", "16")),
    "hnsw_ef_construction": int(os.getenv("HNSW_EF_CONSTRUCTION", "64")),
    "hnsw_ef_search": int(os.getenv("HNSW_EF_SEARCH", "40")),
}

# Create NEW vector store instance
vector_store = PGVectorStore.from_params(
    database=pg_database,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="llama_index_docs",  # Your table
    embed_dim=384,
    hybrid_search=True,
    text_search_config="english",
    hnsw_kwargs=HNSW_KWARGS
)

# DEBUG: Print what table is actually being used
print("=" * 60)
print("VECTOR STORE CONFIGURATION:")
print("=" * 60)
print(f"Table name: {vector_store.table_name}")
print(f"Schema: {vector_store.schema_name}")
print(f"Hybrid search: {vector_store.hybrid_search}")
print("=" * 60)

# Create index from existing vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)


def get_phoenix_prompt(prompt_name: str) -> str:
    """Fetch prompt from Arize Phoenix with detailed logging"""
    logger.info(f"Attempting to fetch prompt '{prompt_name}' from Phoenix...")
    logger.info(f"Phoenix API URL: {PHOENIX_API_URL}")
    
    try:
        url = f"{PHOENIX_API_URL}/v1/prompts/{prompt_name}/latest"
        logger.debug(f"Full request URL: {url}")
        
        response = requests.get(url, timeout=5)
        
        logger.debug(f"Response status code: {response.status_code}")
        
        prompt_data = response.json()
        logger.debug(f"Response JSON keys: {list(prompt_data.keys())}")
        
        data = prompt_data["data"]
        logger.debug(f"Data : {list(data)}")
        
        template = data["template"]
        logger.debug(f"Template type: {template.get('type')}")
        
        messages = template["messages"]
        logger.debug(f"Found {len(messages)} messages in template")
        
        system_message = messages[0]
                              # Extract text from content array
        text_content = [
            item["text"] 
            for item in system_message["content"] 
            if item.get("type") == "text"
        ]
                                
        if text_content:
            template_text = "\n".join(text_content)
            logger.info(f"Successfully loaded prompt '{prompt_name}' from Phoenix")
            logger.info(f"Raw Phoenix prompt content: {template_text}")
            logger.info(f"Prompt length: {len(template_text)} characters")
            logger.info(f"Prompt preview: {template_text[:150]}...")
            return template_text
                    
                
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: Cannot connect to Phoenix at {PHOENIX_API_URL}")
        logger.error(f"Details: {str(e)}")
        logger.info("Make sure Phoenix is running: phoenix serve")
        return None
        


def fusion_hybrid_search(query, vector_limit=10, bm25_limit=10, top_k=5):
    """Hybrid search without requiring OpenAI"""
    global index
    try:
        # Vector retriever
        vector_retriever = index.as_retriever(
            vector_store_query_mode="default",
            similarity_top_k=vector_limit
        )
        
        # Text retriever
        text_retriever = index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=bm25_limit
        )

   
        
        # Fusion retriever with NO LLM requirement
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, text_retriever],
            llm=llama_index_llm_,  # ← Explicitly disable LLM
            similarity_top_k=top_k,
            num_queries=1,
            mode="relative_score",
            use_async=False,
        )
        
        results = fusion_retriever.retrieve(query)
        
        # Convert to your format
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.node.id_,
                'text': result.node.text,
                'contextual_text': result.node.metadata.get('contextual_text', ''),
                'combined_text': result.node.metadata.get('combined_text', ''),
                'metadata': result.node.metadata.get('metadata_', {}),
                'file_name': result.node.metadata.get('file_name', ''),
                'score': result.score if result.score else 0.0,
                'source': 'hybrid_fusion'
            })
        
        # logger.info(f"Fusion Hybrid Results= {formatted_results}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in fusion hybrid search: {e}")
        return []
        

def rerank_results(query, results, top_k=5):
    """Rerank results using a cross-encoder model"""
    if not results:
        return []
    
    try:
        # Load cross-encoder model
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Prepare query-document pairs using combined text
        pairs = []
        for result in results:
            text = result.get('combined_text', result.get('text', ''))
            if text:  # Ensure text is not None or empty
                pairs.append([query, text])
            else:
                pairs.append([query, "No content available"])
        
        # Get relevance scores
        scores = model.predict(pairs)
        
        # Add scores to results
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        # logger.info(f"Reranked results, {reranked_results[:top_k]} ")
        
        return reranked_results[:top_k]
        
    except Exception as e:
        logger.error(f"Error in reranking: {e}")
        return results[:top_k]

def hybrid_search_with_reranking(query: str, vector_limit: int = 10, bm25_limit: int = 10, final_limit: int = 3):
    """Perform hybrid search with reranking for best results"""
    logger.info(f"Hybrid Search with Reranking: '{query}'")
    logger.info(f"Parameters: vector_limit={vector_limit}, bm25_limit={bm25_limit}, final_limit={final_limit}")
    

    final_results = fusion_hybrid_search(query, vector_limit=vector_limit, bm25_limit=bm25_limit, top_k=final_limit)
    
    # Rerank the results
    logger.info("Reranking results...")
    reranked_results = rerank_results(query, final_results, top_k=final_limit)
    
    # Add reranking information
    for result in reranked_results:
        result['source'] = 'hybrid_reranked'
    
    logger.info(f"Returning {len(reranked_results)} reranked results")
    return reranked_results




# Global variables for CrewAI agent
_captured_contexts = []
_captured_scores = []
llm = None


   
llm = LLM(
    model= "ollama/qwen2.5:7b", #"mistral:7b-instruct-q2_K",
    base_url="http://localhost:11434",
    api_key="ollama",  # Ollama doesn't require a real API key
    temperature=0.1,
    context_window=8192,

    # max_tokens=256
)
logger.info("CrewAI LLM setup with Ollama (qwen2.5:7b) successful")

@tool
def hybrid_search_tool(query: str) -> str: #(query: Union[str, Dict[str, Any]]) -> str:
    """Search the knowledge base using hybrid retrieval.
    
    Args:
        query: The search query string to find relevant information
    """
    global _captured_contexts, _captured_scores
    
    # Handle different input formats from CrewAI
    if isinstance(query, dict):
        logger.info(f"Query dict received: {query}")
        query = query.get('query', query.get('description', str(query)))
    elif hasattr(query, 'query'):
        logger.info(f"Query object received: {query}")
        query = query.query
    elif hasattr(query, 'description'):
        logger.info(f"Query object with description received: {query}")
        query = query.description
    
    logger.info(f"Hybrid search tool received query: '{query}'")
    query = str(query).strip()
    
    if not query or query == 'None' or query == '':
        return "Error: No valid search query provided."
    
    # logger.info(f"Hybrid search tool searching for: '{query}'")
    
    try:
        # Use our hybrid search with reranking
        results = hybrid_search_with_reranking(query, vector_limit=20, bm25_limit=20, final_limit=5)
        
        if not results:
            return "No relevant information found."
        
        # Capture contexts and scores for later use
        _captured_contexts = [result.get('text', '') for result in results]
        _captured_scores = [result.get('score', 0.0) for result in results]
        
        result_text = f"Found {len(results)} sources:\n\n"
        for i, result in enumerate(results, 1):
            result_text += f"Source {i} (Score: {result.get('score', 0.0):.3f}):\n"
            result_text += f"File: {result.get('file_name', 'Unknown')}\n"
            result_text += f"{result.get('text', '')}\n\n"
        
        logger.info(f"Hybrid search tool retrieved {len(results)} documents")
        logger.info(f"Result text: {result_text}")
        return result_text
        
    except Exception as e:
        logger.error(f"Error in hybrid search tool: {e}")
        return f"Error searching knowledge base: {str(e)}"


def complete_retrieval_with_agent(query, conversation_history, top_k=5, verbose=True ):
  
    global _captured_contexts, _captured_scores, USE_OLLAMA, llm
    
    llm_provider = "Ollama (qwen2.5:7b)" if USE_OLLAMA else "OpenAI (gpt-4o-mini)"
    logger.info(f"Starting CrewAI agent retrieval for: '{query}' using {llm_provider}")

    logger.info(f"Query parameter received: '{query}'")
    logger.info(f"Query type: {type(query)}")
    logger.info(f"Query length: {len(query)} characters")
    
    # Reset captured data
    _captured_contexts = []
    _captured_scores = []
    
    if not llm:
        logger.error("Cannot proceed without LLM")
        return {
            "query": query,
            "answer": "Error: LLM not available",
            "sources": [],
            "contexts": [],
            "scores": [],
            "num_results": 0,
            "method": "crewai_agent_retrieval"
        }
    
    # Fetch task description from Phoenix
    logger.info("Fetching task description from Phoenix...")
    phoenix_task_description = get_phoenix_prompt("rag_analyst")
    
    # Use Phoenix prompt if available, otherwise use default
    if phoenix_task_description:
        logger.info("Using task description from Phoenix")
        logger.info(f"riginal Phoenix prompt: {phoenix_task_description}")
        
        
        # Replace question placeholder
        task_description = phoenix_task_description.replace("{question}", query)
        logger.info(f"Final task description after question replacement: {task_description}")
        logger.info(f"Task description length: {len(task_description)} characters")
    else:
        logger.warning("Phoenix prompt not available - using default task description")
        task_description = f"Use the hybrid_search_tool to search for information about: '{query}'. Then provide a clear, direct answer based on the search results. Do not ask follow-up questions or suggest additional topics."
        logger.info(f"Default task description: {task_description}")
    
    # Create RAG Analyst Agent
    analyst = Agent(
        role="Retrieval Augmented Generative (RAG) Analyst",
        goal="Answer ONLY the user's question using retrieved results from the hybrid_search_tool tool.",
        backstory=(
            "Expert at information retrieval. Search once, answer clearly."
    ),
        tools=[hybrid_search_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=3,
        max_retry_limits=2,
        max_execution_time=180,
        # reasoning=True,
        # max_reasoning_attempts=1

    )
    
    # Create Task
    task = Task(
        description=f"""
        Previous Conversation: {conversation_history}
        
        Current Query: {query}
          

        {task_description}
        """,
        expected_output="""A professional answer to the user's question based on retrieved information.
                        Steps:
                        1. Search once using: {"query": "search query"}
                        2. Use the results to answer the question
                        3. Provide to the point answer and cite source file names at the end.

                        Do NOT search multiple times with the same query.""",
        agent=analyst
    )
    
    # Create and run Crew
    logger.info("Starting CrewAI execution")
    # Get the current file's directory (i.e., agentic_rag/app/api)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level (to agentic_rag/app)
    crewai_storage_dir = os.path.dirname(current_dir)
    
    os.environ["CREWAI_STORAGE_DIR"] = crewai_storage_dir

    

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=verbose,
        respect_context_window=True,

    #     memory=True,
    #     embedder={
    #     "provider": "ollama",
    #     "config": {
    #         "model": "mxbai-embed-large",  # or "nomic-embed-text"'
    #     }
    # }
        
    )
    
    try:
        result = crew.kickoff(inputs={"question": query})
        logger.info("CrewAI execution completed successfully")
        
        return {
            "query": query,
            "answer": str(result),
            # "sources": [f"Source {result.get('file_name', 'Unknown')}" for i in range(len(_captured_contexts))],
            "contexts": _captured_contexts,
            "scores": _captured_scores,
            "num_results": len(_captured_contexts),
            "method": "crewai_agent_retrieval"
        }
        
    except Exception as e:
        logger.error(f"Error in CrewAI processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            "query": query,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "contexts": _captured_contexts,
            "scores": _captured_scores,
            "num_results": len(_captured_contexts),
            "method": "crewai_agent_retrieval"
        }

