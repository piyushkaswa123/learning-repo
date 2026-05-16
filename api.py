from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "hello world"}

class QuestionRequest(BaseModel):
    question: str

 

@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    messages = [HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")]
    response = llm.invoke(messages)
    return {"answer": response.content}