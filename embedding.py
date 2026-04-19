from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
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
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

embeddings = [item.embedding for item in response.data]

sentences = [
    "I love playing cricket",
    "I enjoy watching football",
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "The weather is nice today",
    "I hate playing cricket"
]

# Compare first sentence with all others
for i in range(1, 6):
    similarity = cosine_similarity(embeddings[0], embeddings[i])
    print(f"Similarity between sentence 1 and sentence {i+1}: {similarity:.4f}")