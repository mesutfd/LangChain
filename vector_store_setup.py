from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Sample documents
docs = [
    Document(page_content="We sold 150 apples in March 2024."),
    Document(page_content="In the last 30 days, 430 bananas were sold."),
    Document(page_content="Total revenue from oranges in February 2024 was $3200."),
    Document(page_content="Apple sales dropped 20% compared to January."),
]

# Use local HuggingFace model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Save to Chroma vector store (persistent)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)

print("✅ Local Vector DB initialized and saved!")
if __name__ == "__main__":
    from chromadb import Client
    from chromadb.config import Settings

    client = Client(Settings(persist_directory="./chroma_db"))
    collection = client.get_collection("langchain")

    total_docs = len(collection.get()["ids"])
    print("Total docs:", total_docs)