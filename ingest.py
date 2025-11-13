import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

class RAGIngestionPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_document(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.lower().split('.')[-1]
        
        if ext == 'pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['docx', 'doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()
    
    def ingest_file(self, file_path: str, collection_name: str = "default"):
        documents = self.load_document(file_path)
        chunks = self.text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_DIR
        )
        
        print(f"✅ Ingested {len(chunks)} chunks")
        return vectorstore

def main():
    file_path = input("Enter file path: ")
    pipeline = RAGIngestionPipeline()
    pipeline.ingest_file(file_path)

if __name__ == "__main__":
    main()