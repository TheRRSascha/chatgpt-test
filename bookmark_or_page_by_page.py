import chromadb
from typing import Dict, Tuple, List
import os
import fitz
import PyPDF2
import tiktoken
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


def setup():
    """
    This function sets up the environment variables, encoding, and the collection in the ChromaDB.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Set up the encoding for the OpenAI model
    encoding_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")

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
    collection_chromadb = client.get_or_create_collection(name="chunk_bookmark-test", embedding_function=openai_ef)

    return encoding_tokenizer, collection_chromadb


def get_bookmarks(filepath: str) -> Dict[int, str]:
    """
    This function gets all the bookmarks (table of contents) from the provided PDF file.
    """
    bookmarks_list = {}
    with fitz.open(filepath) as doc:
        toc = doc.get_toc()
        for level, title, page in toc:
            bookmarks_list[page] = title
    return bookmarks_list


def extract_pdf_text_and_pages(file_location: str) -> Dict[int, str]:
    """
    This function extracts text from each page of the provided PDF file.
    """
    with open(file_location, 'rb') as document:
        pdf_reader = PyPDF2.PdfReader(document)
        page_contents = {}
        for page_num, page in enumerate(pdf_reader.pages):
            page_contents[page_num + 1] = page.extract_text()
    return page_contents


def chunk_text(chunk_text_pages: Dict[int, str], chunk_bookmarks: Dict[int, str], max_chunk_limit=1200,
               encodings=None) -> Tuple[List[Tuple[str, int, List[int]]], int, int]:
    """
    This function splits the text into chunks. A new chunk is created when either a bookmarked page is encountered,
    or when adding another page would cause the chunk to exceed the maximum token limit.
    """
    created_chunks = []
    current_chunk = ""
    current_chunk_size = 0
    chunk_num = 1
    valid_chunk_num = 0
    chunk_page_range = []

    for page_num, text in chunk_text_pages.items():
        page_size = len(encodings.encode(text))
        if page_num in chunk_bookmarks or current_chunk_size + page_size > max_chunk_limit:
            if current_chunk_size > 0:
                created_chunks.append((current_chunk.strip(), current_chunk_size, chunk_page_range))
                valid_chunk_num += 1
                chunk_num += 1
            current_chunk = text
            current_chunk_size = page_size
            chunk_page_range = [page_num]
        else:
            current_chunk += " " + text
            current_chunk_size += page_size
            chunk_page_range.append(page_num)

        if page_num in chunk_bookmarks and len(created_chunks) >= 2:
            chunk_1, chunk_1_size, chunk_1_pages = created_chunks[-2]
            chunk_2, chunk_2_size, chunk_2_pages = created_chunks[-1]
            if chunk_1_size + chunk_2_size <= max_chunk_limit:
                combined_chunk = chunk_1 + " " + chunk_2
                combined_chunk_pages = chunk_1_pages + chunk_2_pages
                created_chunks[-2] = (combined_chunk, chunk_1_size + chunk_2_size, combined_chunk_pages)
                del created_chunks[-1]

    if current_chunk:
        created_chunks.append((current_chunk.strip(), current_chunk_size, chunk_page_range))

    return created_chunks, chunk_num, valid_chunk_num


def print_summary(pdf_file, text_pages, num_chunks, valid_chunks, bookmarks):
    """
    This function prints a summary of the processed PDF file.
    """
    print(f"File: {pdf_file}")
    print(f"Total number of pages: {len(text_pages)}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Number of valid chunks: {valid_chunks}")
    print(f"Bookmarks: {bookmarks}\n")
    print("*" * 90)


def process_pdf_files(files, l_encoding, g_collection):
    """
    This function processes a list of PDF files. For each file, it extracts bookmarks, splits the text into chunks,
    and adds the chunks to the ChromaDB collection.
    """
    for pdf_file in files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        bookmarks = get_bookmarks(pdf_path)
        text_pages = extract_pdf_text_and_pages(pdf_path)
        chunks, num_chunks, valid_chunks = chunk_text(text_pages, bookmarks, max_chunk_limit=1250, encodings=l_encoding)
        print_summary(pdf_file, text_pages, num_chunks, valid_chunks, bookmarks)

        metadatas = [{"source": pdf_file, "tokens": chunk_size} for chunk, chunk_size, _ in chunks]
        g_collection.add(
            documents=[chunk for chunk, _, _ in chunks],
            metadatas=metadatas,
            ids=[f"{os.path.splitext(pdf_file)[0]}_chunk_{i + 1}.pdf" for i in range(len(chunks))]
        )


pdf_folder = "pdf_files"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

encoding, collection = setup()
process_pdf_files(pdf_files, encoding, collection)
