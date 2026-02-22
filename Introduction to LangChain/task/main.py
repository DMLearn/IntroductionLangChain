from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import dotenv
import os
from pathlib import Path

dotenv.load_dotenv()

# Read planet files and store in dictionary
planets_folder = Path("planets")
planet_data = {}
for file_path in planets_folder.glob("*.txt"):
    planet_name = file_path.stem
    with open(file_path, 'r', encoding='utf-8') as f:
        planet_data[planet_name] = f.read().strip()

# Create Chroma client and collection
client = chromadb.Client()

# Create documents with metadata for each planet
documents = [
    Document(page_content=content, metadata={"planet": planet_name})
    for planet_name, content in planet_data.items()
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")),
    client=client,
    collection_name="planets"
)

print(vectorstore.similarity_search(input())[0].page_content)
