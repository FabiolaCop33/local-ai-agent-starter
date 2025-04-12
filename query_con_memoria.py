from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 🧠 1. Configurar los embeddings para recuperar contexto desde Chroma
embeddings = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory="embeddings", embedding_function=embeddings)

# 🤖 2. Cargar el modelo local (Mistral) a través de Ollama
llm = Ollama(model="mistral")

# 🧾 3. Crear memoria de conversación (guarda preguntas y respuestas previas)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔁 4. Crear agente RAG con memoria
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    verbose=True
)

# 🧪 5. Bucle interactivo para hacer preguntas al agente con contexto
print("💬 Agente con memoria activado. Escribe 'salir' para terminar.\n")

while True:
    query = input("Tú: ")
    if query.lower() in ["salir", "exit", "quit"]:
        print("👋 Hasta luego.")
        break
    query = "Eres un asistente que responde solo en español, de forma clara y concisa. " + query
    result = qa_chain.run(query)
    print("🤖:", result)

