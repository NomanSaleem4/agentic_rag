"""
Multi-Agent RAG FastAPI Server with Phoenix Tracing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import time
import uuid
import os
import time
from loguru import logger
from dotenv import load_dotenv

# Import new retrieval pipeline
from app.pipeline.retrieval import complete_retrieval_with_agent

load_dotenv()


ENABLE_TRACING = True  # Set to False to disable tracing

if ENABLE_TRACING:
    try:
        from phoenix.otel import register
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor  # Add this

        from opentelemetry import trace as trace_api
        
        # Setup Phoenix
        tracer_provider = register(
            project_name="contextual-rag-system",
            endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        )
        
        tracer = trace_api.get_tracer(__name__)
        
        # Instrument
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)  # Add this


        logger.info("Phoenix tracing enabled")
    except Exception as e:
        logger.warning(f"Phoenix tracing not available: {e}")
        ENABLE_TRACING = False
        tracer = None
else:
    tracer = None



app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "multi-agent-rag"
    messages: List[Message]
    temperature: Optional[float] = 0.7


class ChatResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatResponseChoice]


N_MESSAGES = 4  # Number of previous messages to include in context

# API ENDPOINTS


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """Chat completion endpoint (OpenAI compatible)"""

    # logger.info(f"Received chat: {len(request.messages)} \n\n")
    
    # Get user message
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    conversation_history = []
    count = 0
    
    for msg in reversed(request.messages):
        # Skip the current message (we'll use it as query)
        if count == 0 and msg.role == "user":
            count += 1
            continue
            
        conversation_history.insert(0, {
            "role": msg.role,
            "content": msg.content
        })
        
        count += 1
        if count >= N_MESSAGES + 1:  # +1 because we skip current
            break
    
    # Format history as readable string
    formatted_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history
    ])    
    
    logger.info(f"Filtered conversation history (N={N_MESSAGES}):")
    logger.info(f"{formatted_history}\n")
    logger.info(f"Current query: {user_message}\n")

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    logger.info(f"\nProcessing: {user_message}\n")

    
    # Process with optional tracing
    if ENABLE_TRACING and tracer:
            
        try:  
            # Use CrewAI agent for retrieval
            result = complete_retrieval_with_agent(user_message, formatted_history, top_k=3, verbose=True)
            answer = result["answer"]
        except Exception as e:
            logger.error(f"Error during question processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            result = complete_retrieval_with_agent(user_message, top_k=3, verbose=True)
            answer = result["answer"]
        except Exception as e:
            logger.error(f"Error during question processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Return OpenAI-compatible response
    return ChatResponse(
        id=f"chat-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatResponseChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop"
            )
        ]
    )


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{"id": "multi-agent-rag", "object": "model", "created": int(time.time())}]
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "tracing_enabled": ENABLE_TRACING,
        "pipeline": "crewai_agent_retrieval",
        "phoenix_url": "http://localhost:6006" if ENABLE_TRACING else None
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)