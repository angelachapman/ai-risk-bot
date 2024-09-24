from operator import itemgetter
import os
from typing import cast
import json

import chainlit as cl

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

import logging
logging.basicConfig(level=logging.INFO)

# Import your fancy_rag_chain function
from vars import CHILD_CHUNK_SIZE, CHILD_OVERLAP, GPT_4O, LOCATION, SYSTEM_PROMPT_TEMPLATE, TE3_LARGE, TE3_VECTOR_LENGTH


# Initialize ChatOpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
    
def init_retriever ():
    def read_docs_from_file(filename):
        """Reads a list of Langchain documents from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return [Document(**doc) for doc in data]

    docs = read_docs_from_file("chunked_docs.json")

    # Qdrant client and vectorstore
    qdrant_client = QdrantClient(location=LOCATION) 
    qdrant_client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=TE3_VECTOR_LENGTH, distance=Distance.COSINE),
    )
    print('created qdrant client')
    qdrant_fulldoc_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="my_collection",
        embedding=OpenAIEmbeddings(model=TE3_LARGE),
    )

    # Create the new retriever
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_OVERLAP)
    parentdoc_retriever = ParentDocumentRetriever(
        vectorstore=qdrant_fulldoc_vector_store,
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
    )
    print("created retriever")
    return parentdoc_retriever,docs

@cl.on_chat_start
async def start():    
    # Initialize the RAG chain
    parentdoc_retriever, docs = init_retriever()

    print('adding docs to vector db')
    await parentdoc_retriever.aadd_documents(docs)
    print('populated vector db')

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    primary_qa_llm = ChatOpenAI(model_name=GPT_4O, temperature=0, streaming=True)

    rag_chain = (
        {"context": itemgetter("input") | parentdoc_retriever, "input": itemgetter("input")} 
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | primary_qa_llm}
    )
    print('created chain')
    
    cl.user_session.set("chain",rag_chain)

    msg = cl.Message(content="I'm ready to chat! My expertise is in AI and how it's regulated. How can I help you today?")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    chain = cast(Runnable, cl.user_session.get("chain"))  # type: Runnable
    if not chain: print("chain not found in session")
    else: print("retrieved chain")

    msg = cl.Message(content="")

    try:
        async for chunk in chain.astream(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            print(f"received chunk {chunk}")
            await msg.stream_token(chunk["response"].content)
    except Exception as e:
        print(f"Error in chain execution: {e}")
        msg.content = "An error occurred processing your request"

    await msg.send()

if __name__ == "__main__":
    cl.run()