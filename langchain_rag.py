from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("PIYUSH_AI_DE.pdf")  # replace with your filename
pages = loader.load()

# print(f"Number of pages: {len(pages)}")
# print(f"First page content: {pages[0].page_content[:200]}")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(pages)

# print(f"Number of chunks: {len(chunks)}")
# print(f"First chunk: {chunks[0].page_content}")


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory="./chroma_langchain"
# )

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain"
)

# print(f"Stored {vectorstore._collection.count()} chunks in ChromaDB")

question = "what are piyush's skills?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_chunks = retriever.invoke(question)

context = "\n".join([doc.page_content for doc in relevant_chunks])
# print(f"Retrieved context:\n{context}")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = f"Based on this context:\n{context}\n\nAnswer this question: {question}"

response = llm.invoke(prompt)
print(response.content)