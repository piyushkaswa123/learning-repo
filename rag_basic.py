from dotenv import load_dotenv
import os
import chromadb
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="vectoreStore")


user_question = "which sport is more popular?"

# Step 1: embed the question
response2 = client.embeddings.create(
    model="text-embedding-3-small",  # fill this
    input=[user_question]
)



# Step 2: query chromadb for top 2 results
results = collection.query(
    query_embeddings=[item.embedding for item in response2.data],  # embedding of your search query
    n_results=2           # return top 2 most similar
)

# Step 3: pass results as context to GPT like this:

context = "\n".join(results['documents'][0])
prompt = f"Based on this context: {context}\n\nAnswer this: {user_question}"

# Step 4: call OpenAI chat API with that prompt
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
])

# Step 5: print the answer
print(response.choices[0].message.content)