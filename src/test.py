from retrieval import load_all_docs
docs = load_all_docs()
s02e17 = [d for d in docs if d.metadata['season'] == 2 and d.metadata['episode'] == '17']
for d in s02e17:
    if 'dub' in d.page_content.lower() or 'not me' in d.page_content.lower() or 'voice' in d.page_content.lower():
        print(d.page_content[:500])
        print("---")