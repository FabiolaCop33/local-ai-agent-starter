# ⚠️ Proyecto Finalizado Temporalmente

Este repositorio implementa un agente de IA local utilizando:

- LangChain 🧠  
- Ollama (modelo mistral) 🦙  
- ChromaDB 🗃️  
- Streamlit (interfaz estilo chat) 💬

## ✅ Fases completadas

✔️ Fase 1: Preparación del entorno (entorno virtual, dependencias, estructura de carpetas)  
✔️ Fase 2: Indexación de conocimiento y embeddings con `mistral` vía Ollama  
✔️ Fase 3: Construcción del agente conversacional con memoria  
✔️ Fase 4: Pruebas funcionales vía consola  
✔️ Fase 5: Interfaz visual tipo WhatsApp con Streamlit

## ❌ Problema detectado

Durante las pruebas finales se descubrió que **Ollama fue instalado vía Homebrew**, lo cual:

- Solo instala el cliente CLI (`ollama`)  
- **No incluye ni activa el servicio Ollama** (daemon que escucha en `localhost:11434`)  
- Impide que LangChain o cualquier script pueda consultar modelos como `mistral` localmente

## 💡 Solución recomendada

1. Desinstalar Ollama instalado por Homebrew:
   ```bash
   brew uninstall ollama

