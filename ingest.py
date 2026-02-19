from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PERSIST_DIR = "./chroma_db"

def ingest_documents():
    loader = PyPDFLoader("documents/Sample.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    print("✅ Documents embedded successfully!")

if __name__ == "__main__":
    ingest_documents()
