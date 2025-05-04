import os
import time
import requests

from chromadb import HttpClient
from typing import List
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Connect to the remote Chroma running on Docker
chroma_client = HttpClient(host="localhost", port=8000)

load_dotenv()
# Set a directory to persist the vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# persist_directory = "chroma_store"

documents = [
    Document(page_content="We sold 150 KG apples in March 2024."),
    Document(page_content="In the last 30 days, 430 bananas were sold. Today is December 13 from 2024."),
    Document(page_content="Total revenue from oranges in February 2024 was $3200."),
    Document(page_content="Today weather is rainy in Paris."),
    Document(page_content="Apple sales dropped 20% compared to January. This is for September 2024"),
    Document(page_content="Apple company stocks fall $80000 in last month. Today is Mach 30."),
    Document(page_content="Our company sales are growing by selling 1000 KG apples. Today is 30 March 2024."),
    Document(page_content="Apple price is $3 per KG."),
    Document(page_content="Amazon is biggest online shop in the world."),
    Document(page_content="In Cargo table, LoadFromDate is Cargo Issuance DateTime, Not CreatedAt."),
]

time1 = time.time()
vectorstore = Chroma(
    client=chroma_client,
    collection_name="langchain",
    embedding_function=embedding
)


def filter_new_documents(vector: Chroma, new_docs: List[Document]) -> List[Document]:
    # Get existing docs (you may paginate this if many)
    existing = vector.get()
    existing_contents = set(existing['documents'])

    # Filter out docs with same page_content
    unique_docs = [doc for doc in new_docs if doc.page_content not in existing_contents]
    return unique_docs


def insert_chroma_document(vector: Chroma, doc_list: List[Document]):
    return vectorstore.add_documents(doc_list)


unique_doc_list = filter_new_documents(vector=vectorstore, new_docs=documents)

if unique_doc_list:
    insert_chroma_document(vector=vectorstore, doc_list=unique_doc_list)

time2 = time.time()


def enrich_prompt(query: str, vectorstore: Chroma = vectorstore, k: int = 5) -> str:
    """
    Retrieve relevant documents from Chroma vectorstore and create an enriched prompt for the model.

    Args:
    - query: The user's question or prompt.
    - vectorstore: The Chroma vector store to search.
    - k: Number of relevant documents to retrieve (default 5).

    Returns:
    - enriched_prompt: The context-enriched prompt for the model.
    """
    retriever = vectorstore.as_retriever(search_type="similarity", k=k)
    docs = retriever.get_relevant_documents(query, k=k)

    # Collect the relevant documents
    context = "\n".join([doc.page_content for doc in docs])

    # Create the enriched prompt
    enriched_prompt = f"Context:\n{context}\n\nUser Question:\n{query}"

    return enriched_prompt


t3 = time.time()
query = "Estimate revenue from selling apples in March."
enriched_prompt = enrich_prompt(query, vectorstore, k=5)
print("=" * 36)
print(enriched_prompt)
print("=" * 36)

print(f"embedding time: {time2 - time1}s")
print(f"query time: {t3 - time2}s")


def ask_deepseek(prompt: str, api_key: str = os.getenv("DEEPSEEK_API_KEY")) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"  # Replace with the actual URL if different

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an error if the request failed

    return response.json()["choices"][0]["message"]["content"]

# result = ask_deepseek(prompt=enriched_prompt)
# print(result)
