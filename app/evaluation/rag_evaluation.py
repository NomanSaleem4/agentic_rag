import os
import sys
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall 
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
from loguru import logger
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


from app.pipeline.retrieval import complete_retrieval_with_agent

load_dotenv()
EVAL_DATASET_PATH = os.path.join(project_root, 'app', 'evaluation', 'evaluation_data.json')

conversation_history = ""  # Initialize empty conversation history

def run_rag_pipeline(query):

    result = complete_retrieval_with_agent(query=query, conversation_history=conversation_history, top_k=5, verbose=False)
    answer = result.get('answer', 'No answer generated')
    contexts = result.get('contexts', [])
    
    if contexts and isinstance(contexts[0], dict):
        contexts = [ctx.get('text', ctx.get('combined_text', '')) for ctx in contexts]
    
    contexts = [str(ctx) for ctx in contexts if ctx and str(ctx).strip()]
    if not contexts and answer:
        contexts = [answer]
    
    logger.info(f"Answer: {answer}")
    logger.info(f"Contexts: {contexts}")
    return {"answer": answer, "contexts": contexts}
 

def evaluate_rag():

    logger.info(f"Loading evaluation data from: {EVAL_DATASET_PATH}")
    with open(EVAL_DATASET_PATH, 'r') as f:
        eval_data = json.load(f)
    
    eval_data = eval_data[:15]  # Test with 5 questions
    logger.info(f"Evaluating {len(eval_data)} questions")
 
    ollama_llm = ChatOllama(
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=120
    )
    ollama_embeddings = OllamaEmbeddings(
        model="all-minilm:l6-v2",
        base_url="http://localhost:11434"
    )
    ragas_llm = LangchainLLMWrapper(ollama_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
    logger.info("Ollama configured")
    
    
    questions = [item['question'] for item in eval_data]
    ground_truths = [item['ground_truth'] for item in eval_data]
    results = []
    
    for i, q in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] {q[:60]}...")
        start_time = time.time()

        result = run_rag_pipeline(q)
        
        elapsed = time.time() - start_time
        logger.info(f"Done in {elapsed:.2f}s | Contexts: {len(result['contexts'])}")
        results.append(result)
    
    logger.info(f"\n Pipeline complete!\n")
    
    # Prepare dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": ground_truths,
    })
    
    
    metrics = [
        faithfulness,      # Is answer factual based on context?
        answer_relevancy,  # Does answer address the question?
        context_recall
    ]
    
    start_time = time.time()
    result = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False
    )
    elapsed = time.time() - start_time
    
    logger.info(f"\n Evaluation done in {elapsed:.2f}s!")
    logger.info(f"\n{result}\n")
    
    # Save results
    result_dict = result.to_pandas().to_dict()
    with open('ragas_results.json', 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info("Saved to ragas_results.json")
    
    # Summary
    
    df = result.to_pandas()
    for col in ['faithfulness', 'answer_relevancy', 'context_recall']:
        if col in df.columns:
            mean_val = df[col].mean()
            logger.info(f"  {col}: {mean_val:.4f}")
        

if __name__ == "__main__":
    evaluate_rag()