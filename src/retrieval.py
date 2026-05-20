import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.ingest import load_episode, split_into_scenes, parse_filename
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def load_all_docs():
    scripts_dir = "data/scripts"
    all_docs = []

    for filename in sorted(os.listdir(scripts_dir)):
        if not filename.endswith(".html"):
            continue
        filepath = os.path.join(scripts_dir, filename)
        season, episode = parse_filename(filename)
        text = load_episode(filepath)
        scenes = split_into_scenes(text)

        for i, scene in enumerate(scenes):
            if len(scene.strip()) < 100:
                continue
            all_docs.append(Document(
                page_content=scene.strip()[:5000],
                metadata={
                    "season": season,
                    "episode": episode,
                    "scene_index": i,
                    "source": filename
                }
            ))

    return all_docs

def rerank(query, docs, top_n=3, model=None):
    if model is None:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(float(score), doc) for score, doc in scored_docs[:top_n]]

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_results):
        key = doc.page_content[:50]
        scores[key] = scores.get(key, 0) + 1/(rank + k)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:50]
        scores[key] = scores.get(key, 0) + 1/(rank + k)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]

def compare_retrievers(query, k=5):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    all_docs = load_all_docs()
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # vector only
    vector_results = db.similarity_search(query, k=k)
    print(f"\n--- Vector Search ---")
    for i, doc in enumerate(vector_results):
        print(f"[{i+1}] S{doc.metadata['season']:02d}E{doc.metadata['episode']} | {doc.page_content[:100]}...")

    # BM25 only
    bm25_results = bm25_retriever.invoke(query)
    print(f"\n--- BM25 Search ---")
    for i, doc in enumerate(bm25_results):
        print(f"[{i+1}] S{doc.metadata['season']:02d}E{doc.metadata['episode']} | {doc.page_content[:100]}...")

    # hybrid via RRF
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)
    print(f"\n--- Hybrid Search (RRF) ---")
    for i, doc in enumerate(hybrid_results[:5]):
        print(f"[{i+1}] S{doc.metadata['season']:02d}E{doc.metadata['episode']} | {doc.page_content[:100]}...")

if __name__ == "__main__":
    queries = [
        "How you doin",
        "Smelly Cat",
        "character dealing with loneliness",
        "friends supporting each other",
    ]

    for query in queries:
        compare_retrievers(query)

    print("\n\n=== RERANKING TEST ===")
    query = "Smelly Cat"

    # get top 10 from hybrid
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    all_docs = load_all_docs()
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10

    vector_results = db.similarity_search(query, k=10)
    bm25_results = bm25_retriever.invoke(query)
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)[:10]

    print(f"\nBefore reranking (top 5 of 10):")
    for i, doc in enumerate(hybrid_results[:5]):
        print(f"[{i+1}] S{doc.metadata['season']:02d}E{doc.metadata['episode']} | {doc.page_content[:80]}...")

    print(f"\nAfter reranking (top 3):")
    reranked = rerank(query, hybrid_results, top_n=3)
    for score, doc in reranked:
        print(f"Score: {score:.4f} | S{doc.metadata['season']:02d}E{doc.metadata['episode']} | {doc.page_content[:80]}...")