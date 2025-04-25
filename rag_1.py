from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import openai
from langchain_qdrant import QdrantVectorStore

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Load a PDF file using PyPDFLoader
pdf_path = Path(__file__).parent / "book1.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)

# Create embeddings for the split documents
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)

# Uncomment this to inject data only once
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="book1",
    embedding=embedder
)
print("Documents added to vector store.")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="book1",
    embedding=embedder
)

query = input(">> ")

relevant_chunk = retriever.similarity_search(
    query=query
)
print(f"\n\n\n\n\nRelevant chunk: {relevant_chunk} \n\n\n\n\n")

# Prepare context from page_content of each document
context_text = "\n\n".join([doc.page_content for doc in relevant_chunk])

# Inject context into system prompt
system_prompt = f"""
You are a helpful AI Assistant who responds to the user queries based on the context provided.

Context:
{context_text}
"""

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Call the OpenAI API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
)

# Print the assistant's reply
print(response)
