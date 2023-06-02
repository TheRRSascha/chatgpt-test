import chromadb
from typing import Dict, Tuple, List
import os
import fitz
import PyPDF2
import tiktoken
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the encoding for the OpenAI model
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")

# Create the embedding function for OpenAI's text-embedding-ada-002 model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

# Create a ChromaDB client and collection
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/",
    anonymized_telemetry=False
))

# Get or create the collection for storing text chunks
collection = client.get_or_create_collection(name="chunk_bookmark-test", embedding_function=openai_ef)


def get_bookmarks(filepath: str) -> Dict[int, str]:
    bookmarks_list = {}
    with fitz.open(filepath) as doc:
        toc = doc.get_toc()
        for level, title, page in toc:
            bookmarks_list[page] = title
    return bookmarks_list


def extract_pdf_text_and_pages(file_location: str) -> Tuple[str, Dict[int, str]]:
    with open(file_location, 'rb') as document:
        pdf_reader = PyPDF2.PdfReader(document)
        page_contents = {}
        for page_num, page in enumerate(pdf_reader.pages):
            page_contents[page_num + 1] = page.extract_text()  # +1 to make page numbers 1-based
    return page_contents


def chunk_text(chunk_text_pages: Dict[int, str], chunk_bookmarks: Dict[int, str],  max_chunk_limit=1200,
               ) -> Tuple[List[str], int, int]:
    created_chunks = []
    current_chunk = ""
    current_chunk_size = 0
    chunk_num = 1
    valid_chunk_num = 0
    chunk_page_range = []

    for page_num, text in chunk_text_pages.items():
        page_size = len(encoding.encode(text))  # use tokenizer for encoding

        if page_num in chunk_bookmarks or current_chunk_size + page_size > max_chunk_limit:
            if current_chunk_size > 0:
                created_chunks.append((current_chunk.strip(), current_chunk_size, chunk_page_range))
                print(
                    f"Chunk {chunk_num}: pages {chunk_page_range[0]}-{chunk_page_range[-1]}, tokens: {current_chunk_size}")
                valid_chunk_num += 1
                chunk_num += 1
            current_chunk = text
            current_chunk_size = page_size
            chunk_page_range = [page_num]
        else:
            current_chunk += " " + text
            current_chunk_size += page_size
            chunk_page_range.append(page_num)

        # After processing a bookmarked page, check if the last two chunks can be combined
        if page_num in chunk_bookmarks and len(created_chunks) >= 2:
            chunk_1, chunk_1_size, chunk_1_pages = created_chunks[-2]
            chunk_2, chunk_2_size, chunk_2_pages = created_chunks[-1]
            # If combined chunk size is within limit, combine chunks
            if chunk_1_size + chunk_2_size <= max_chunk_limit:
                combined_chunk = chunk_1 + " " + chunk_2
                combined_chunk_pages = chunk_1_pages + chunk_2_pages
                created_chunks[-2] = (combined_chunk, chunk_1_size + chunk_2_size, combined_chunk_pages)
                del created_chunks[-1]
                print(
                    f"Combined chunk: pages {combined_chunk_pages[0]}-{combined_chunk_pages[-1]}, tokens: {chunk_1_size + chunk_2_size}")

    # Add the last chunk if it's not empty
    if current_chunk:
        created_chunks.append((current_chunk.strip(), current_chunk_size, chunk_page_range))
        print(f"Chunk {chunk_num}: pages {chunk_page_range[0]}-{chunk_page_range[-1]}, tokens: {current_chunk_size}")

    return created_chunks, chunk_num, valid_chunk_num


pdf_folder = "C:\\PDFs Testfiles"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    bookmarks = get_bookmarks(pdf_path)
    text_pages = extract_pdf_text_and_pages(pdf_path)
    chunks, num_chunks, valid_chunks = chunk_text(text_pages, bookmarks, max_chunk_limit=1250)
    print(f"File: {pdf_file}")
    print(f"Total number of pages: {len(text_pages)}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Number of valid chunks: {valid_chunks}")
    print(f"Bookmarks: {bookmarks}\n")
    print("*" * 90)

    # Create metadatas list
    metadatas = []
    for chunk, chunk_size, _ in chunks:
        metadata = {"source": pdf_file, "tokens": chunk_size}
        metadatas.append(metadata)

"""   # Pass metadatas list to add method
    collection.add(
        documents=[chunk for chunk, _, _ in chunks],
        metadatas=metadatas,
        ids=[f"{os.path.splitext(pdf_file)[0]}_chunk_{i + 1}.pdf" for i in range(len(chunks))]
    )"""
