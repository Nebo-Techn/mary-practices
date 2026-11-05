from pathlib import Path

import chromadb
import requests
from ollama import chat


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def retrieve_context(query: str, n_results: int = 5) -> str:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []
    for chunk, metadata in zip(chunks, metadatas):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        context_parts.append(f"[Source: {source}, page {page}]\n{chunk}")
        # print(context_parts)

    return "\n\n".join(context_parts)


def ask_budget_ai(question: str) -> str:
    context = retrieve_context(question)
    if not context:
        context = "No relevant budget information was found in the knowledge base."

    prompt = f"""You are Budget AI, a helpful assistant for Tanzanian budget documents.
Answer the user's question only from the context below. If the context does not contain the answer, say so clearly. Be concise and cite the source and page when possible.

Context:
{context}

Question:
{question}

Answer:"""
    response = chat(
        model="llama2",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return response["message"]["content"]


    # response = requests.post(
    #     OLLAMA_URL,
    #     json={
    #         "model": MODEL_NAME,
    #         "prompt": prompt,
    #         "stream": False,
    #     },
    #     timeout=120,
    # )
    # response.raise_for_status()
    # return response.json()["response"]


def main() -> None:
    print("Budget AI")
    print("Ask a question about the budget documents. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            answer = ask_budget_ai(question)
            print(f"Budget AI: {answer}\n")
        except requests.RequestException as error:
            print(f"Budget AI: Could not reach Ollama ({error}). Make sure Ollama is running.\n")
        except KeyError as error:
            print(f"Budget AI: Ollama returned an unexpected response (missing {error}).\n")


if __name__ == "__main__":
    main()
