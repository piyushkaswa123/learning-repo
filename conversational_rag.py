from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("PIYUSH_AI_DE.pdf")  # replace with your filename
pages = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(pages)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain"
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-3.5-turbo")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chat_history = []

print("Ask questions about the document. Type 'quit' to exit.")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    
    # Get relevant chunks
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Build messages with history
    messages = [HumanMessage(content=f"Context: {context}\n\nChat history: {chat_history}\n\nQuestion: {question}")]
    
    response = llm.invoke(messages)
    print(f"Bot: {response.content}\n")
    
    # Store history
    chat_history.append({"user": question, "bot": response.content})