import asyncio
from operator import itemgetter

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


LOCATION = ":memory:"
VECTOR_SIZE = 1536
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful expert. Answer user questions based only on the context below. 
If you don't know, say nah.

Question:
{input}

Context:
{context}
"""

async def load_and_chunk_pdf(pdf:str, chunk_size:int, chunk_overlap:int) -> list[str]:
    """Load a pdf file, combine it into one doc, split it, and return the chunks"""
    print(f"Loading {pdf}...")
    pages = await PyMuPDFLoader(file_path=pdf).aload()

    print("Chunking...")
    combined_text = "\n".join([doc.page_content for doc in pages])
    combined_document = Document(page_content=combined_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the combined document
    return await text_splitter.atransform_documents([combined_document])

# Function to do vanilla RAG on a bunch of text strings that are already chunked
async def vanilla_rag_chain(texts:list[Document], openai_key:str, collection_name:str):

    qdrant_client = AsyncQdrantClient(location=LOCATION) 
    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print('created qdrant client')

    embeddings = OpenAIEmbeddings( model=EMBEDDING_MODEL)
    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings  # Embedding function from OpenAI embeddings
    )
    await qdrant_vector_store.aadd_documents(texts)
    retriever = qdrant_vector_store.as_retriever()
    print('populated vector db')

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    primary_qa_llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("input") | retriever, "input": itemgetter("input")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
    )
    print('created chain')

    return retrieval_augmented_qa_chain

