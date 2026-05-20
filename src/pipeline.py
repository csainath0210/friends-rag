import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

def load_pipeline(k=5):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

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

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain

def ask(question, k=5):
    chain = load_pipeline(k=k)
    result = chain.invoke({"query": question})

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['result']}")
    print(f"\n--- Retrieved Scenes ---")
    for i, doc in enumerate(result['source_documents']):
        print(f"\n[{i+1}] S{doc.metadata['season']:02d}E{doc.metadata['episode']} | Scene {doc.metadata['scene_index']}")
        print(doc.page_content)
        print("...")

if __name__ == "__main__":
    questions = [
        "What happens when someone eats Ross's sandwich at work?",
    ]

    for q in questions:
        ask(q)