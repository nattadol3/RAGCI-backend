import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.schema.document import Document
from get_embedded import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
pdf_path = "pdf_data"
image_path = 'image_data' # Must specify path by user
urls = [
    'https://www.en.kku.ac.th/web/%E0%B8%87%E0%B8%B2%E0%B8%99%E0%B8%9A%E0%B8%A3%E0%B8%B4%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%A7%E0%B8%B4%E0%B8%8A%E0%B8%B2%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B9%81%E0%B8%A5%E0%B8%B0%E0%B8%A7%E0%B8%B4%E0%B8%88/#1523875822874-a039c957-3a3f',
    'https://www.en.kku.ac.th/web/1-%E0%B8%A3%E0%B8%B2%E0%B8%87%E0%B8%A7%E0%B8%B1%E0%B8%A5%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%95%E0%B8%B5%E0%B8%9E%E0%B8%B4%E0%B8%A1%E0%B8%9E%E0%B9%8C%E0%B9%80%E0%B8%9C%E0%B8%A2%E0%B9%81%E0%B8%9E/',
    'https://www.en.kku.ac.th/web/2-%E0%B8%A3%E0%B8%B2%E0%B8%87%E0%B8%A7%E0%B8%B1%E0%B8%A5%E0%B8%9C%E0%B8%A5%E0%B8%87%E0%B8%B2%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B9%84%E0%B8%94%E0%B9%89/',
    'http://www.en.kku.ac.th/web/3-%E0%B8%A3%E0%B8%B2%E0%B8%87%E0%B8%A7%E0%B8%B1%E0%B8%A5%E0%B8%9C%E0%B8%A5%E0%B8%87%E0%B8%B2%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B8%AB%E0%B8%A3%E0%B8%B7%E0%B8%AD%E0%B8%87%E0%B8%B2/',
    'http://www.en.kku.ac.th/web/4-%E0%B8%A3%E0%B8%B2%E0%B8%87%E0%B8%A7%E0%B8%B1%E0%B8%A5%E0%B8%AA%E0%B8%99%E0%B8%B1%E0%B8%9A%E0%B8%AA%E0%B8%99%E0%B8%B8%E0%B8%99%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%88%E0%B8%B1%E0%B8%94%E0%B8%97%E0%B8%B3/',
    'http://www.en.kku.ac.th/web/5-%E0%B8%A3%E0%B8%B2%E0%B8%87%E0%B8%A7%E0%B8%B1%E0%B8%A5%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%AD%E0%B9%89%E0%B8%B2%E0%B8%87%E0%B8%AD%E0%B8%B4%E0%B8%87%E0%B8%9A%E0%B8%97%E0%B8%84%E0%B8%A7%E0%B8%B2/',
    'http://www.en.kku.ac.th/web/6-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B9%80%E0%B8%9E%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B8%A3%E0%B9%88%E0%B8%A7%E0%B8%A1%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%99%E0%B8%B3/',
    'http://www.en.kku.ac.th/web/7-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B8%94%E0%B9%89%E0%B8%B2%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%8A%E0%B8%B2%E0%B8%A7%E0%B8%B4%E0%B8%A8%E0%B8%A7%E0%B8%81/',
    'http://www.en.kku.ac.th/web/8-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B9%80%E0%B8%9E%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B9%80%E0%B8%AA%E0%B8%A3%E0%B8%B4%E0%B8%A1%E0%B8%A8/',
    'http://www.en.kku.ac.th/web/9-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%AA%E0%B8%A1%E0%B8%97%E0%B8%9A%E0%B9%82%E0%B8%84%E0%B8%A3%E0%B8%87%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B8%97%E0%B8%B8%E0%B8%99/',
    'http://www.en.kku.ac.th/web/10-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%81%E0%B8%A5%E0%B8%B8%E0%B9%88%E0%B8%A1%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2/',
    'http://www.en.kku.ac.th/web/11-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B8%AA%E0%B8%96%E0%B8%B2%E0%B8%9A%E0%B8%B1%E0%B8%99/',
    'http://www.en.kku.ac.th/web/12-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%A1%E0%B8%B5%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%A3%E0%B9%88%E0%B8%A7%E0%B8%A1/',
    'http://www.en.kku.ac.th/web/13-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%AD%E0%B8%B8%E0%B8%94%E0%B8%AB%E0%B8%99%E0%B8%B8%E0%B8%99%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%84%E0%B9%89%E0%B8%99%E0%B8%84%E0%B8%A7%E0%B9%89%E0%B8%B2%E0%B9%81/',
    'http://www.en.kku.ac.th/web/14-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B9%80%E0%B8%9E%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%9E%E0%B8%B1%E0%B8%92%E0%B8%99%E0%B8%B2%E0%B8%95%E0%B9%89%E0%B8%99/',
    'http://www.en.kku.ac.th/web/15-%E0%B8%97%E0%B8%B8%E0%B8%99%E0%B8%A7%E0%B8%B4%E0%B8%88%E0%B8%B1%E0%B8%A2%E0%B9%80%E0%B8%9E%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%AA%E0%B9%88%E0%B8%87%E0%B9%80%E0%B8%AA%E0%B8%A3%E0%B8%B4%E0%B8%A1/',
    'https://rad.kku.ac.th/Menu_support/publication.php',
    'https://rad.kku.ac.th/Menu_support/presentation.php'
]

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    pdf_docs = load_documents()
    image_docs = load_image_document()
    url_docs = load_url_document()
    documents = pdf_docs + url_docs + image_docs
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(pdf_path)
    return document_loader.load()

def load_url_document():
    web_loader = WebBaseLoader(urls)
    web_docs = web_loader.load()
    return web_docs

def load_image_document():
    loader = UnstructuredImageLoader(image_path)
    image_data = loader.load()
    return image_data

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()