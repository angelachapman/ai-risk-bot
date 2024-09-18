from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_chunk_pdf(pdf:str, chunk_size:int, chunk_overlap:int) -> list[str]:
    """Load a pdf file, combine it into one doc, split it, and return the chunks"""
    print(f"Loading {pdf}...")
    pages = PyMuPDFLoader(file_path=pdf).load()

    print("Chunking...")
    combined_text = "\n".join([doc.page_content for doc in pages])
    combined_document = Document(page_content=combined_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the combined document
    return text_splitter.split_documents([combined_document])