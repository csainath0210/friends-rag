import os
from bs4 import BeautifulSoup

def load_episode(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n")

def split_into_scenes(text):
    lines = text.split("\n")
    scenes = []
    current_scene = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[Scene:") or line.startswith("[scene:"):
            if current_scene:
                scenes.append("\n".join(current_scene))
            current_scene = [line]
        else:
            current_scene.append(line)

    if current_scene:
        scenes.append("\n".join(current_scene))

    return scenes

def parse_filename(filename):
    # e.g. 0101.html -> season 1, episode 1
    name = filename.replace(".html", "")
    season = int(name[:2])
    episode = name[2:]
    return season, episode

if __name__ == "__main__":
    scripts_dir = "data/scripts"
    all_chunks = []

    for filename in sorted(os.listdir(scripts_dir)):
        if not filename.endswith(".html"):
            continue

        filepath = os.path.join(scripts_dir, filename)
        season, episode = parse_filename(filename)
        text = load_episode(filepath)
        scenes = split_into_scenes(text)

        for i, scene in enumerate(scenes):
            all_chunks.append({
                "text": scene,
                "metadata": {
                    "season": season,
                    "episode": episode,
                    "scene_index": i,
                    "source": filename
                }
            })

    print(f"Total episodes processed: {len(os.listdir(scripts_dir))}")
    print(f"Total scenes extracted: {len(all_chunks)}")
    print(f"\n--- Sample scene ---\n")
    print(all_chunks[2]["text"])
    print(f"\nMetadata: {all_chunks[2]['metadata']}")