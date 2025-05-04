from chromadb import HttpClient

client = HttpClient(host="192.168.21.76", port=8000)
collection = client.get_collection("langchain")

print(client.list_collections())
# Get all IDs and count
doc_count = len(collection.get()['ids'])
print(f"Total documents: {doc_count}")

