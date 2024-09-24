import asyncio
from operator import itemgetter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyMuPDFLoader

import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from datasets import Dataset

from vars import LOCATION, OPENAI_VECTOR_SIZE, HF_VECTOR_SIZE, SYSTEM_PROMPT_TEMPLATE, TEST_DATASET_FILE
from vars import PARENT_CHUNK_SIZE, PARENT_OVERLAP, CHILD_CHUNK_SIZE, CHILD_OVERLAP
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def gen_rag_responses(rag_chain) -> Dataset:
    """Wrapper function to run a RAG chain against a test dataset and generate/store responses"""
    test_df = pd.read_csv(TEST_DATASET_FILE)

    test_questions = test_df["question"].to_list()
    test_gt = test_df["ground_truth"].to_list()
    print("read test questions")

    answers = []
    contexts = []

    print("generating responses")
    for question in tqdm_asyncio(test_questions,desc="Processing Questions"):
        response = await rag_chain.ainvoke({"input" : question})
        answers.append(response["response"].content)
        contexts.append([context.page_content for context in response["context"]])

    # Put in huggingface dataset format and save it for later re-use
    response_dataset = Dataset.from_dict({
        "question" : test_questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : test_gt
    })

    return response_dataset

async def load_and_chunk_pdf(pdf:str, chunk_size:int, chunk_overlap:int) -> list[Document]:
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
                            collection_name:str="default",
                            vector_size:int = OPENAI_VECTOR_SIZE ):
    """Construct a RAG chain using Qdrant and a specified set of OpenAI models"""

    qdrant_client = QdrantClient(location=LOCATION) 
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
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

# Function to do RAG on a bunch of text strings that are already chunked,
# with pre-defined embeddings
async def vanilla_rag_chain_hf_embeddings( texts:list[Document], 
                                            openai_key:str, # for the chat model
                                            embeddings:HuggingFaceEmbeddings, 
                                            chat_model:str, 
                                            collection_name:str="default_hf" ):
    """Construct a RAG chain using Qdrant, a fine-tuned embedding model, and 
    an OpenAI chat model. Could easily be combined with vanilla_openai_rag_chain, with a bit
    of simple refactoring"""

    qdrant_client = QdrantClient(location=LOCATION) 
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=HF_VECTOR_SIZE, distance=Distance.COSINE),
    )
    print('created qdrant client')
    
    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings  
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

async def fancy_rag_chain(pdf_dict:dict, # should have file_path, skip_pages_begin (int or None), skip_pages_end (int or None)  
                        openai_key:str, # for the chat model
                        embeddings:HuggingFaceEmbeddings, 
                        chat_model:str, 
                        collection_name:str="default_fancy",
                        use_streaming:bool=False  ):
    
    """Load pdf files, discarding irrelevant front and back material. Split them into a parent/child
    structure. Contstruct and return a RAG pipeline. As before, could easily be combined with other 
    code if we did a bit of refactoring.
    
    Note that, unlike our RAG chain functions above, this one also does the doc loading"""
    docs = []
    for key, value in pdf_dict.items():
        skip_pages_begin = value.get("skip_pages_begin")
        skip_pages_end = value.get("skip_pages_end")
    
        # Load the PDF using PyMuPDFLoader
        print(f"loading {value["file_path"]}")
        doc = PyMuPDFLoader(value["file_path"]).load()

        if skip_pages_begin is not None: doc = doc[skip_pages_begin:]
        if skip_pages_end is not None: doc = doc[:-skip_pages_end]

        docs.extend(doc)    

    print(f"Loaded {len(docs)} docs")

    # Define parent and child splitters
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_OVERLAP)

    # Qdrant client and vectorstore
    qdrant_client = QdrantClient(location=LOCATION) 
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=HF_VECTOR_SIZE, distance=Distance.COSINE),
    )
    print('created qdrant client')

    qdrant_fulldoc_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Create the new retriever
    parentdoc_retriever = ParentDocumentRetriever(
        vectorstore=qdrant_fulldoc_vector_store,
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
    )

    await parentdoc_retriever.aadd_documents(docs)
    print('populated vector db')

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    primary_qa_llm = ChatOpenAI(model_name=chat_model, temperature=0, streaming=use_streaming)

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("input") | parentdoc_retriever, "input": itemgetter("input")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
    )
    print('created chain')

    return retrieval_augmented_qa_chain

