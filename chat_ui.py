import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Configuración inicial
st.set_page_config(page_title="Agente IA - Más por Menos", page_icon="🤖")
st.title("🤖 Agente IA - Más por Menos Club")

# Inicializar memoria y carga si no existe
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# Cargar base vectorial
embeddings = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory="embeddings", embedding_function=embeddings)
llm = Ollama(model="mistral")

# Crear agente conversacional
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=st.session_state.memory,
    verbose=False  # Ocultamos logs técnicos
)

# Entrada del usuario
user_input = st.text_input("Haz tu pregunta 👇", placeholder="Ej. ¿Qué es Mas por Menos Club?")

# Mostrar conversación
if user_input:
    with st.spinner("Pensando..."):
        response = qa_chain.invoke({"question": user_input})
        st.markdown(f"**🤖 Respuesta:** {response}")
