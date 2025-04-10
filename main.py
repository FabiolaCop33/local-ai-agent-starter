from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Ruta base del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(CURRENT_DIR, "docs")

# Cargar documentos .txt desde la carpeta docs
def load_documents():
    docs = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_PATH, filename))
            docs.extend(loader.load())
    return docs

# Fragmentar los documentos en trozos manejables
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Guardar embeddings en Chroma
def index_documents(chunks):
    embeddings = OllamaEmbeddings(model="mistral")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="embeddings")
    print("✅ Documentos indexados y guardados.")

# Flujo principal
if __name__ == "__main__":
    print("🔍 Cargando documentos...")
    documents = load_documents()
    print(f"📄 {len(documents)} documentos cargados.")

    print("✂️ Fragmentando documentos...")
    chunks = split_documents(documents)
    print(f"🧩 {len(chunks)} fragmentos creados.")

    print("📦 Indexando en ChromaDB...")
    index_documents(chunks)

