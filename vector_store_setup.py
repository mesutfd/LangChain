from chromadb import HttpClient

client = HttpClient(host="localhost", port=8000)
collection = client.get_collection("langchain")

print(client.list_collections())
# Get all IDs and count
doc_count = len(collection.get()['ids'])
print(f"Total documents: {doc_count}")

