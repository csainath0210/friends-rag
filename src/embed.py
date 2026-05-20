import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_episode, split_into_scenes, parse_filename
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def build_vectorstore():
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
            # skip very short scenes (stage directions, etc)
            if len(scene.strip()) < 100:
                continue
            # truncate long scenes instead of skipping them
            scene_text = scene.strip()[:1500]
            all_docs.append(Document(
                page_content=scene_text,
                metadata={
                    "season": season,
                    "episode": episode,
                    "scene_index": i,
                    "source": filename
                }
            ))

    print(f"Total docs to embed: {len(all_docs)}")
    print("Embedding... this will take a few minutes")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        all_docs,
        embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Done. Vectorstore saved to ./chroma_db")
    return db

if __name__ == "__main__":
    build_vectorstore()