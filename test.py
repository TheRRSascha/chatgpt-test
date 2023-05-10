import os
import pandas as pd
import PyPDF2
import chromadb
from chromadb.config import Settings
import tiktoken
from chromadb.utils import embedding_functions
import nltk
from dotenv import load_dotenv
load_dotenv()

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key= os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-ada-002"
            )

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/"
))

collection = client.get_or_create_collection(name="test",embedding_function=openai_ef)

def extract_pdf_text(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, max_chunk_size=1000):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_chunk_size = 0
    chunk_num = 1

    for sentence in sentences:
        sentence_size = len(nltk.word_tokenize(sentence))
        if current_chunk_size + sentence_size > max_chunk_size and current_chunk_size > 0:
            chunks.append(current_chunk.strip())
            print(f"Chunk {chunk_num} ({current_chunk_size} tokens)")
            current_chunk = ""
            current_chunk_size = 0
            chunk_num += 1

        current_chunk += sentence + " "
        current_chunk_size += sentence_size

    if current_chunk_size > 0:
        chunks.append(current_chunk.strip())
        print(f"Chunk {chunk_num} ({current_chunk_size} tokens)")
        chunk_num += 1

    return chunks, chunk_num - 1

df = pd.DataFrame(columns=['fileName', 'content', 'tokens'])

pdf_folder = "C:\\PDFs Testfiles"
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file_name)
        content = extract_pdf_text(file_path)
        chunks, num_chunks = chunk_text(content, max_chunk_size=1000)
        tokens = sum(len(encoding.encode(chunk)) for chunk in chunks)
        df.loc[len(df)] = [file_name, chunks, tokens]
        print(f"File: {file_name}")
        print(f"Number of chunks: {num_chunks}")
        print(f"Total tokens: {tokens}\n")
        total_chunks = df['content'].apply(len).sum()
        print(f"Total chunks of all PDFs: {total_chunks}")

print(df)

for index, row in df.iterrows():
    file_name = row['fileName']
    chunks = row['content']
    tokens = row['tokens']

    # Add the chunks to the ChromaDB collection
    collection.add(
        documents=chunks,
        metadatas=[{"source": file_name, "tokens": tokens}] * len(chunks),
        ids=[f"{os.path.splitext(file_name)[0]}_chunk_{i+1}.pdf" for i in range(len(chunks))]
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
    print(results["metadatas"])
    print(results["ids"])

    print("Top 3 results:")
    for i, result in enumerate(results["documents"]):
        for j, chunk in enumerate(result):  # Iterate over each chunk in the result
            print(f"Chunk {j + 1}:")  # Print "Chunk 1", "Chunk 2", etc.
            print("*" * 50)  # Print a row of asterisks before each chunk
            print(chunk)  # Print the chunk on a separate line
            print("*" * 50)  # Print a row of asterisks after each chunk
            print("\n")

print("Exiting...")
