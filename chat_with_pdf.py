import os
import chromadb
from chromadb.config import Settings
import requests
import json

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import openai

load_dotenv()
# set your api key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-ada-002"
            )

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/"
))

collection = client.get_collection(name="test", embedding_function=openai_ef)

while True:
    question = input("Enter your question (or type 'exit' to quit): ")
    print("*" * 90)
    if question.lower() == "exit":
        break
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }

    data = {
        'input': question,
        'model': 'text-embedding-ada-002',
    }

    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, data=json.dumps(data))

    # The json method returns the json-encoded content of a response, if any.
    json_response = response.json()
    # Print the json response
    # print(json_response)

    # Save the response to a file
    with open('response.json', 'w') as f:
        json.dump(json_response, f)

    embedding = json_response['data'][0]['embedding']

    # Perform ChromaDB query
    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Use the information provided to the assistant. Add from other sources if needed."},
            {"role": "assistant", "content": results["documents"][0][0]},
            {"role": "user", "content": question}

        ]
    )

    print(chat['choices'][0]['message']['content'])
    print("*" * 90)
    response.close()
print("Exiting....")
