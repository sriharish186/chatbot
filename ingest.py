from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

#dir(PyPDFLoader)
DATA_FOLDER = "data"
VECTOR_DB_PATH = "vectorstore"

documents = []

# Load all PDFs
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

# Chunk text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)
print(f"Created {len(docs)} chunks")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(VECTOR_DB_PATH)

print("Vector store created successfully")