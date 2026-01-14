import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
load_dotenv()

if __name__ == "__main__":
    print("Ingesting....")
    loader = TextLoader("C:/Users/user/langchain_course/mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Splitted into {len(texts)} chunks")
    
    embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY"))
    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME")
    )
    print("Done")
    
    
    