import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage
from src.retrieval import load_all_docs, reciprocal_rank_fusion, rerank
from src.agent import build_agent
from sentence_transformers import CrossEncoder
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load once at startup
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
all_docs = load_all_docs()
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 10
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

prompt_template = """You are a helpful assistant that answers questions about the TV show Friends.
Use ONLY the following script excerpts to answer the question.
If the answer is not in the excerpts, say "I don't know".

Script excerpts:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

class QuestionRequest(BaseModel):
    question: str

def expand_query(question: str, llm) -> list[str]:
    prompt = f"""Generate 2 alternative search queries for this question that use different words but mean the same thing. Return only the queries, one per line, no numbering.

    Question: {question}

    Alternative queries:"""
    response = llm.invoke(prompt)
    alternatives = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return [question] + alternatives[:2]

@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question

    # time each step
    t0 = time.perf_counter()
    vector_results = db.similarity_search(question, k=10)
    t1 = time.perf_counter()

    bm25_results = bm25_retriever.invoke(question)
    t2 = time.perf_counter()

    queries = expand_query(question, llm)
    print(f"Expanded queries: {queries}")

    all_vector = []
    all_bm25 = []
    for q in queries:
        all_vector.extend(db.similarity_search(q, k=5))
        all_bm25.extend(bm25_retriever.invoke(q))

    hybrid_results = reciprocal_rank_fusion(all_vector, all_bm25)[:10]
    reranked = rerank(question, hybrid_results, top_n=3, model=reranker_model)
    t3 = time.perf_counter()

    top_docs = [doc for _, doc in reranked]
    context_text = "\n\n".join([doc.page_content for doc in top_docs])
    full_prompt = prompt_template.replace("{context}", context_text).replace("{question}", question)

    response = llm.invoke(full_prompt)
    t4 = time.perf_counter()

    return {
        "answer": response.content,
        "retrieved_chunks": [
            {
                "text": doc.page_content,
                "score": float(score),
                "episode": doc.metadata.get("episode"),
                "season": doc.metadata.get("season"),
                "source": doc.metadata.get("source")
            }
            for score, doc in reranked
        ],
        "metrics": {
            "vector_search_ms": round((t1 - t0) * 1000),
            "bm25_search_ms": round((t2 - t1) * 1000),
            "rerank_ms": round((t3 - t2) * 1000),
            "llm_ms": round((t4 - t3) * 1000),
            "total_ms": round((t4 - t0) * 1000)
        }
    }

@app.post("/agent/ask")
def agent_ask(request: QuestionRequest):
    agent = build_agent()
    steps = []

    # patch to capture steps
    original_invoke = agent.invoke

    result = agent.invoke({
        "messages": [HumanMessage(content=request.question)],
        "steps_taken": 0,
        "max_steps": 10
    })

    # extract tool calls from messages
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "type": "tool_call",
                    "tool": tc["name"],
                    "input": tc["args"]
                })
        elif hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            # tool results
            if hasattr(msg, "name"):
                steps.append({
                    "type": "tool_result",
                    "tool": msg.name,
                    "output": msg.content[:300]
                })

    final_answer = result["messages"][-1].content

    return {
        "answer": final_answer,
        "steps": steps,
        "total_steps": result["steps_taken"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}