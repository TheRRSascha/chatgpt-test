# Import the required libraries and modules
import datetime
import os
import pandas as pd
import PyPDF2
import chromadb
from chromadb.config import Settings
import tiktoken
from chromadb.utils import embedding_functions
import nltk
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

# Create a ChromaDB client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/",
    anonymized_telemetry=False
))

# Get or create the collection for storing text chunks
collection = client.get_or_create_collection(name="chunk_sentence-experimental", embedding_function=openai_ef)

log_file = "droppedchunks/dropped_chunks_" + datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".txt"


def extract_pdf_text(file_location):
    """
    Extract text from a PDF file.

    Args:
        file_location (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    with open(file_location, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Define a function to chunk the text
def chunk_text(text, max_chunk_size=1100, max_chunk_limit=1200, name_file="unknown"):
    """
    Reducing the text into smaller chunks.

    Args:
        text (str): The text to be chunked.
        max_chunk_size (int): The maximum size of each chunk in tokens.
        max_chunk_limit (int): The maximum size limit for a valid chunk in tokens.
        name_file (str): The name of the file being chunked.

    Returns:
        list: The list of created chunks.
        int: The total number of chunks.
        int: The number of valid chunks.
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    created_chunks = []
    current_chunk = ""
    current_chunk_size = 0
    chunk_num = 1
    valid_chunk_num = 0

    for sentence in sentences:
        sentence_size = len(encoding.encode(sentence))
        if current_chunk_size + sentence_size > max_chunk_size:
            if current_chunk_size > 0:
                if current_chunk_size > max_chunk_limit:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{name_file} | Chunk {chunk_num} | {current_chunk_size} tokens\n{current_chunk.strip()}\n")
                    print(f"Chunk {chunk_num} ({current_chunk_size} tokens) dropped due to size limit.")
                else:
                    created_chunks.append(current_chunk.strip())
                    valid_chunk_num += 1
                chunk_num += 1
            current_chunk = sentence
            current_chunk_size = sentence_size
        else:
            current_chunk += " " + sentence
            current_chunk_size += sentence_size

    if current_chunk:
        if current_chunk_size > max_chunk_limit:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{name_file} | Chunk {chunk_num} | {current_chunk_size} tokens\n{current_chunk.strip()}\n")
            print(f"Chunk {chunk_num} ({current_chunk_size} tokens) dropped due to size limit.")
        else:
            created_chunks.append(current_chunk.strip())
            valid_chunk_num += 1

    return created_chunks, chunk_num, valid_chunk_num


# Create a DataFrame to store the chunked data
df = pd.DataFrame(columns=['fileName', 'content', 'tokens'])

pdf_folder = "pdf_files"  # Define the path to the folder containing the PDF files
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file_name)
        content = extract_pdf_text(file_path)
        chunks, num_chunks, valid_chunks = chunk_text(content, max_chunk_size=1100, name_file=file_name)
        tokens = sum(len(encoding.encode(chunk)) for chunk in chunks)
        df.loc[len(df)] = [file_name, chunks, tokens]
        print(f"File: {file_name}")
        print(f"Number of chunks: {num_chunks}")
        print(f"Number of valid chunks: {valid_chunks}")
        print(f"Total tokens: {tokens}\n")
        total_chunks = df['content'].apply(len).sum()
        print(f"Total chunks of all PDFs: {total_chunks}")


print(df)

# Add the chunks to the ChromaDB collection
for index, row in df.iterrows():
    file_name = row['fileName']
    chunks = row['content']
    tokens = row['tokens']

    collection.add(
        documents=chunks,
        metadatas=[{"source": file_name, "tokens": tokens}] * len(chunks),
        ids=[f"{os.path.splitext(file_name)[0]}_chunk_{i + 1}.pdf" for i in range(len(chunks))]
    )


while True:
    query_text = input("Enter your search query (or type 'exit' to quit): ")

    if query_text.lower() == "exit":
        break

    # Perform ChromaDB query
    results = collection.query(
        query_texts=query_text,
        n_results=2
    )

    print(results)

    print("Top results:")
    for i, result in enumerate(results["documents"]):
        for j, chunk in enumerate(result):
            print(f"Chunk {j + 1}:")
            print("*" * 50)
            print(chunk)
            print("*" * 50)
            print("\n")

print("Exiting...")
