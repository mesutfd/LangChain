import time
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Set a directory to persist the vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# persist_directory = "chroma_store"

documents = [
    Document(page_content="We sold 150 apples in March 2024."),
    Document(page_content="In the last 30 days, 430 bananas were sold."),
    Document(page_content="Total revenue from oranges in February 2024 was $3200."),
    Document(page_content="Today weather is rainy in Paris."),
    Document(page_content="Apple sales dropped 20% compared to January."),
    Document(page_content="Apple company stocks fall $80000 in last month."),
]

t1 = time.time()

# Use persistent Chroma
vectorstore = Chroma.from_documents(documents, embedding)  # pass this param to persist: persist_directory=persist_directory
#vectorstore.persist()  # <-- This saves the data to disk, uncomment it
t2 = time.time()

# Later, you can load it using:
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectorstore.as_retriever(search_type="similarity", k=3)

query = "how many sells did we have from selling apples last month?"
docs = retriever.get_relevant_documents(query)
t3 = time.time()

for i, doc in enumerate(docs, 1):
    print(f"Chunk {i}:\n{doc.page_content}\n")

context = "\n".join([doc.page_content for doc in docs])
enriched_prompt = f"Context:\n{context}\n\nUser Question:\n{query}"

print(enriched_prompt)
print(f"embedding time: {t2 - t1}s")
print(f"query time: {t3 - t2}s")
