import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.retrieval import load_all_docs, reciprocal_rank_fusion, rerank
from src.ingest import load_episode, parse_filename
import os

# load once at module level — not on every tool call
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
all_docs = load_all_docs()
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 10

@tool
def search_scripts(query: str, season: int = None) -> str:
    """Search the Friends scripts for scenes relevant to the query.
    Optionally filter by season number (1-10).
    Returns the most relevant scenes with episode information."""

    vector_results = db.similarity_search(query, k=10)
    bm25_results = bm25_retriever.invoke(query)
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)[:10]

    # filter by season if specified
    if season is not None:
        # handle case where LLM passes a list instead of int
        if isinstance(season, list):
            season = season[0] if season else None
        
        hybrid_results = [
            doc for doc in hybrid_results
            if doc.metadata.get("season") == season
        ]

    reranked = rerank(query, hybrid_results[:10], top_n=3)

    output = []
    for score, doc in reranked:
        s = doc.metadata['season']
        e = doc.metadata['episode']
        output.append(f"[S{s:02d}E{e}] (relevance: {score:.2f})\n{doc.page_content[:500]}")

    return "\n\n---\n\n".join(output) if output else "No relevant scenes found."

@tool
def get_character_episodes(character_name: str) -> str:
    """Get all episodes that feature a specific character.
    Returns a list of episodes mentioning this character."""

    character_upper = character_name.upper()
    matching = []

    for doc in all_docs:
        if character_upper in doc.page_content.upper():
            ep = f"S{doc.metadata['season']:02d}E{doc.metadata['episode']}"
            if ep not in matching:
                matching.append(ep)

    if not matching:
        return f"No episodes found featuring {character_name}"

    return f"Episodes featuring {character_name}: {', '.join(sorted(matching))}"

@tool
def get_episode_info(season: int, episode: int) -> str:
    """Get information about a specific Friends episode.
    Returns the opening scenes and location of the episode."""

    filename = f"{season:02d}{episode:02d}.html"
    filepath = os.path.join("data/scripts", filename)

    if not os.path.exists(filepath):
        return f"Episode S{season:02d}E{episode:02d} not found."

    from src.ingest import load_episode, split_into_scenes
    text = load_episode(filepath)
    scenes = split_into_scenes(text)

    # return first 3 scenes as episode overview
    preview = "\n\n".join(scenes[:3])
    return f"S{season:02d}E{episode:02d} opening:\n{preview[:1000]}"