import streamlit as st
from retrieve import RAGRetrievalPipeline

st.set_page_config(page_title="RAG Retrieval", page_icon="📚")

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return RAGRetrievalPipeline()

pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    collection = st.text_input("Collection", value="default")
    use_llm = st.checkbox("Use LLM", value=True)
    top_k = st.slider("Top K", 1, 10, 4)

# Main
st.title("📚 RAG Document Retrieval")

query = st.text_input("Query")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            try:
                result = pipeline.answer_question(query, collection, use_llm, top_k)
                
                if isinstance(result, str):
                    st.write(result)
                elif isinstance(result, list):
                    for i, doc in enumerate(result, 1):
                        st.expander(f"Doc {i}").write(doc.page_content)
            except Exception as e:
                st.error(str(e))

