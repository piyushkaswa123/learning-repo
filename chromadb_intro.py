import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="vectoreStore")  # give it a name

print("Collection created:", collection.name)