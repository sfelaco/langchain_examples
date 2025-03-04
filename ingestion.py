import os
from dotenv import load_dotenv
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


if __name__ == "__main__":
   load_dotenv()
   
   print("Starting the ingestion...")
   loader = TextLoader("mediumblog1.txt", encoding="utf-8")
   document = loader.load()
   
   print("splitting...")
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(document)
   print(f"Created {len(texts)} chunks")
   
   embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get('OPENAI_API_KEY'))
   
   print("ingesting...")
   PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ.get('INDEX_NAME'))
   print("finish")