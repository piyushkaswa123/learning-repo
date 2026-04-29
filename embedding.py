from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb
load_dotenv()
chroma_client= chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="vectoreStore")  # give it a name

print("Collection created:", collection.name)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embeddings.create(
    model="text-embedding-3-small",  # fill this
    input=["I love playing cricket",
"I enjoy watching football",
"Machine learning is fascinating",
"Deep learning uses neural networks",
"The weather is nice today",
"I hate playing cricket"]  # fill this
)




# print(len(response.data[0].embedding))

embeddings = [item.embedding for item in response.data]

sentences = [
    "I love playing cricket",
    "I enjoy watching football",
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "The weather is nice today",
    "I hate playing cricket"
]


collection.add(
    documents=sentences,    # the list of sentences
    embeddings=embeddings,  # the list of vectors
    ids=["id1", "id2", "id3", "id4", "id5", "id6"]
)
response1 = client.embeddings.create(
    model="text-embedding-3-small",  # fill this
    input=["I enjoy sports"]
)
results = collection.query(
    query_embeddings=[item.embedding for item in response1.data],  # embedding of your search query
    n_results=2           # return top 2 most similar
)
print(results)
