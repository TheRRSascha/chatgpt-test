import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/" # Optional, defaults to .chromadb/ in the current directory
))

collection = client.get_collection(name="test")

print(collection.count())
print(collection.peek(10))