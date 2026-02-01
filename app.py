import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_classic.chains.retrieval_qa.base import RetrievalQA



#import langchain_community
#dir(langchain_community)
VECTOR_DB_PATH = "vectorstore"
MODEL_PATH = "models/BioMistral-7B.Q4_K_M.gguf"

st.set_page_config(page_title="IVF Patient Support Chatbot")
st.title("IVF Patient Support Chatbot")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector store
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    temperature=0.1,
    verbose=False
)

# RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = st.text_input("Ask a question about IVF:")

if query:
    with st.spinner("Thinking..."):
        response = qa.invoke({"query": query})["result"]
    st.success("Answer:")
    st.write(response)