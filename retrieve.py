import os
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback for older versions
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import httpx

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
OLLAMA_MODEL = "gpt-oss:20b"

class RAGRetrievalPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _get_vectorstore(self, collection_name: str = "default"):
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )
    
    def search(self, query: str, collection_name: str = "default", top_k: int = 4):
        vectorstore = self._get_vectorstore(collection_name)
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata, "score": score} 
                for doc, score in results]
    
    def answer_question(self, query: str, collection_name: str = "default", use_llm: bool = True, top_k: int = 4):
        vectorstore = self._get_vectorstore(collection_name)
        
        if use_llm:
            try:
                llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize Ollama LLM: {e}")
                print("💡 Make sure Ollama is running: `ollama serve`")
                print("📚 Falling back to context-only retrieval (no LLM generation)...")
                use_llm = False
            
            if use_llm:
                prompt = PromptTemplate.from_template(
                    """You are an expert on A Song of Ice and Fire (Game of Thrones) books. 
Use the following passages from the books to answer the question accurately.
If the answer is not in the context, say you don't know.

Context from the books:
{context}

Question: {question}

Answer:"""
                )
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                try:
                    result = chain.invoke(query)
                    print(f"Answer: {result}")
                    return result
                except httpx.ConnectError as e:
                    print(f"❌ Error: Cannot connect to Ollama server")
                    print(f"💡 Make sure Ollama is running: `ollama serve`")
                    print(f"📚 Falling back to context-only retrieval...")
                    use_llm = False
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "not found" in error_msg.lower():
                        print(f"❌ Error: Model '{OLLAMA_MODEL}' not found")
                        print(f"💡 Available models: Run `ollama list` to see installed models")
                        print(f"💡 Install model: `ollama pull llama3` or `ollama pull mistral`")
                    else:
                        print(f"❌ Error during LLM generation: {e}")
                    print(f"📚 Falling back to context-only retrieval...")
                    use_llm = False
        
        # Fallback: return context documents without LLM
        if not use_llm:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
            # Use invoke() for newer LangChain versions, fallback to get_relevant_documents for older versions
            try:
                documents = retriever.invoke(query)
            except AttributeError:
                # Fallback for older LangChain versions
                documents = retriever.get_relevant_documents(query)
            
            print(f"\n📚 Retrieved {len(documents)} relevant documents:")
            print("=" * 60)
            for i, doc in enumerate(documents, 1):
                print(f"\n📄 Document {i}:")
                print(f"{doc.page_content[:500]}...")
                if doc.metadata:
                    print(f"Metadata: {doc.metadata}")
            print("=" * 60)
            return documents

def main():
    query = input("Enter query: ")
    collection = input("Collection name (default): ") or "default"
    
    retrieval = RAGRetrievalPipeline()
    retrieval.answer_question(query, collection)

if __name__ == "__main__":
    main()