import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 🎨 Estilo de la interfaz
st.set_page_config(page_title="Más por Menos - Agente IA", page_icon="🛒", layout="centered")
st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #D1E9C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛒 Más por Menos Club - Agente IA")
st.caption("Responde preguntas sobre el servicio, planes y más.")

# 🧠 Inicializar memoria
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.history = []

# 📦 Cargar modelo y base vectorial
embeddings = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory="embeddings", embedding_function=embeddings)
llm = Ollama(model="mistral")

# 🔄 Agente con memoria
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=st.session_state.memory,
    verbose=False
)

# 📥 Entrada del usuario
user_input = st.text_input("Pregunta algo sobre Más por Menos...", placeholder="Ej. ¿Qué planes ofrecen?", key="input")

# 💬 Procesar entrada
if user_input:
    with st.spinner("Pensando..."):
        response = qa_chain.invoke({"question": "Responde solo en español. " + user_input})
        st.session_state.history.append(("Tú", user_input))
        st.session_state.history.append(("Agente", response["answer"]))

# 🗂 Mostrar historial de conversación
if st.session_state.history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, msg in st.session_state.history:
        if speaker == "Tú":
            st.markdown(f'<div class="user-bubble"><strong>{speaker}:</strong> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble"><strong>{speaker}:</strong> {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

