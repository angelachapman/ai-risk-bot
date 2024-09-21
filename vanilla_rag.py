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


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful, kind expert in AI safety and risk mitigation. You can answer highly technical
questions, but you also know how to give big-picture answers that laypeople can understand.
Answer user questions based only on the context below. Answer in at least a paragraph and provide lots of
details, but avoid too much jargon.
If you don't know, or if the context is not relevant, just apologize and say "I don't know".

User question:
{input}

Context:
{context}
"""

async def load_and_chunk_pdf(pdf:str, chunk_size:int, chunk_overlap:int) -> list[str]:
    """Load a pdf file, combine it into one doc, split it, and return the chunks"""
    print(f"Loading {pdf}...")
    pages = PyMuPDFLoader(file_path=pdf).load() # aload available in Langchain 0.3

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
async def vanilla_openai_rag_chain( texts:list[Document], 
                            openai_key:str, 
                            embedding_model:str, 
                            chat_model:str, 
                            collection_name:str="default" ):

    qdrant_client = QdrantClient(location=LOCATION) 
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print('created qdrant client')

    embeddings = OpenAIEmbeddings( model=embedding_model )
    print('created embeddings')
    
    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings  # Embedding function from OpenAI embeddings
    )
    await qdrant_vector_store.aadd_documents(texts)
    retriever = qdrant_vector_store.as_retriever()
    print('populated vector db')

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    primary_qa_llm = ChatOpenAI(model_name=chat_model, temperature=0)

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("input") | retriever, "input": itemgetter("input")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
    )
    print('created chain')

    return retrieval_augmented_qa_chain

