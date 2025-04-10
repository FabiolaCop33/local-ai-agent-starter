from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Cargar embeddings y base de datos indexada
embeddings = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory="embeddings", embedding_function=embeddings)

# Cargar modelo LLM
llm = Ollama(model="mistral")

# Crear la cadena RAG (retrieval + generación)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# Hacer una pregunta
query = "¿Qué es Mas por Menos Club?"
result = qa_chain(query)

print("\n🤖 Respuesta:")
print(result["result"])

