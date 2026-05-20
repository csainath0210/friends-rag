import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from src.retrieval import load_all_docs, reciprocal_rank_fusion, rerank
from langchain_community.retrievers import BM25Retriever
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


def avg(lst):
    return sum(lst)/len(lst) if lst else 0

def get_pipeline_answer(question, retriever_type="vector", k=5):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
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

    if retriever_type == "vector":
        retriever = db.as_retriever(search_kwargs={"k": k})
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        result = chain.invoke({"query": question})
        answer = result["result"]
        contexts = [doc.page_content for doc in result["source_documents"]]

    elif retriever_type == "hybrid":
        all_docs = load_all_docs()
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k
        vector_results = db.similarity_search(question, k=k)
        bm25_results = bm25_retriever.invoke(question)
        hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)[:k]
        context_text = "\n\n".join([doc.page_content for doc in hybrid_results])
        full_prompt = prompt_template.replace("{context}", context_text).replace("{question}", question)
        answer = llm.invoke(full_prompt).content
        contexts = [doc.page_content for doc in hybrid_results]

    elif retriever_type == "hybrid_rerank":
        all_docs = load_all_docs()
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 10
        vector_results = db.similarity_search(question, k=10)
        bm25_results = bm25_retriever.invoke(question)
        hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)[:10]
        reranked = rerank(question, hybrid_results, top_n=3)
        top_docs = [doc for _, doc in reranked]
        context_text = "\n\n".join([doc.page_content for doc in top_docs])
        full_prompt = prompt_template.replace("{context}", context_text).replace("{question}", question)
        answer = llm.invoke(full_prompt).content
        contexts = [doc.page_content for doc in top_docs]

    return answer, contexts

def run_evaluation(retriever_type="vector"):
    print(f"\n{'='*60}")
    print(f"Running evaluation: {retriever_type}")
    print(f"{'='*60}")

    with open("evals/golden_dataset.json") as f:
        golden = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(golden):
        print(f"  [{i+1}/{len(golden)}] {item['question'][:50]}...")
        answer, context = get_pipeline_answer(
            item["question"],
            retriever_type=retriever_type
        )
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    judge_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=RunConfig(max_workers=1, timeout=120)
)

    print(f"\nResults for {retriever_type}:")
    print(f"  Context Precision:  {avg(result['context_precision']):.4f}")
    print(f"  Context Recall:     {avg(result['context_recall']):.4f}")
    print(f"  Faithfulness:       {avg(result['faithfulness']):.4f}")
    print(f"  Answer Relevancy:   {avg(result['answer_relevancy']):.4f}")

    return result

if __name__ == "__main__":
    results = {}
    for retriever_type in ["vector", "hybrid", "hybrid_rerank"]:
        results[retriever_type] = run_evaluation(retriever_type)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Vector':>10} {'Hybrid':>10} {'Hybrid+Rerank':>15}")
    print("-"*60)
    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        v = avg(results["vector"][metric])
        h = avg(results["hybrid"][metric])
        hr = avg(results["hybrid_rerank"][metric])
        print(f"{metric:<25} {v:>10.4f} {h:>10.4f} {hr:>15.4f}")