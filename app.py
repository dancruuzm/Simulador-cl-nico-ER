import streamlit as st
import random
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuración de página UI ---
st.set_page_config(page_title="Simulador Clínico Enfermedades Respiratorias", page_icon="🩺", layout="wide")

st.title("🩺 Simulador de Casos Clínicos de Enfermedades Respiratorias")
st.markdown("Pide un paciente, analiza su caso clínico, propón tu diagnóstico y tratamiento, y recibe retroalimentación.")

# --- Inicialización del Sistema RAG ---
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db_v3", embedding_function=embeddings)
    llm = None # Ya no usamos Ollama local, usamos el GPU remoto en evaluate_user
    
    # Este retriever buscará SOLO en las guías clínicas (documentos teóricos)
    retriever_guias = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"tipo": "documento_teorico"}}
    )
    
    return vectorstore, llm, retriever_guias

with st.spinner("Cargando motor de simulación y guías médicas..."):
    vectorstore, llm, retriever_guias = load_rag_system()

# --- Manejo de la Máquina de Estados ---
if "app_state" not in st.session_state:
    st.session_state.app_state = "inicio" # Estados: inicio, evaluacion
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Funciones ---
def get_random_case():
    # Seleccionamos un caso aleatorio de nuestra base de datos de casos (tipo = caso_clinico)
    # Hacemos una búsqueda amplia para traernos varios y escoger uno.
    resultados = vectorstore.similarity_search("paciente", k=50, filter={"tipo": "caso_clinico_real"})
    if resultados:
        return random.choice(resultados)
    return None

def evaluate_user(diagnostico_usuario, caso_real, contexto_guias):
    diagnostico_oculto = caso_real.metadata.get('diagnostico_real', 'Desconocido')
    
    system_prompt = (
        "Eres un tutor médico experto evaluando a un estudiante de medicina.\n\n"
        "=== DIAGNÓSTICO Y EVOLUCIÓN REAL DEL PACIENTE ===\n"
        f"{diagnostico_oculto}\n\n"
        "=== GUÍAS CLÍNICAS OFICIALES (REFERENCIA) ===\n"
        f"{contexto_guias[:2000]}\n\n"
        "=== TU TAREA ===\n"
        "1. Compara el diagnóstico del estudiante con el diagnóstico real. ¿Acertó?\n"
        "2. Evalúa su tratamiento propuesto basándote en las guías clínicas.\n"
        "3. Dale una retroalimentación constructiva, profesional y educativa (máximo 2 o 3 párrafos)."
    )
    
    # --- Conexión al GPU de la UNAM (LM Studio) ---
    import openai
    import base64
    import httpx
    
    USER = 'rag_user'
    PASSWORD = 'plm+cuan-ruf*85735e4a.'
    encoded_credentials = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    
    http_client = httpx.Client(verify=False)
    
    client = openai.OpenAI(
        base_url="https://dinamica1.fciencias.unam.mx/lmstudio/v1/",
        api_key="lm-studio",
        default_headers={
            "Authorization": f"Basic {encoded_credentials}"
        },
        http_client=http_client
    )
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Mi diagnóstico y tratamiento es el siguiente: {diagnostico_usuario}"}
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al conectar con el tutor remoto: {str(e)}"

# --- Interfaz de Pantallas ---

if st.session_state.app_state == "inicio":
    st.info("👋 Bienvenido al Simulador Clínico. Haz clic en el botón para recibir a tu paciente.")
    if st.button("🩺 Asignarme un Paciente", use_container_width=True):
        caso = get_random_case()
        if caso:
            st.session_state.current_case = caso
            st.session_state.app_state = "evaluacion"
            st.session_state.messages = [{"role": "assistant", "content": "De acuerdo con el expediente del paciente. ¿Cuál es su hipótesis diagnóstica? ¿Cuál es el tratamiento sugerido?"}]
            st.rerun()
        else:
            st.error("No se encontraron casos clínicos en la base de datos. Por favor, espera a que termine de ejecutarse el script de ingesta (ingest_spaccc.py).")

elif st.session_state.app_state == "evaluacion":
    caso = st.session_state.current_case
    
    # 1. Panel de Expediente Médico
    with st.expander("📄 **Expediente del Paciente (Activo)**", expanded=True):
        st.write(caso.page_content)
        url_img = caso.metadata.get("url_imagen")
        if url_img and url_img != "nan":
            try:
                # Si el sistema detecta que es el placeholder porque no se bajaron las imágenes reales de Kaggle
                if "fakeimg" in url_img:
                    st.warning("⚠️ Nota: Las imágenes reales de este paciente no se descargaron para ahorrar espacio. Mostrando radiografía de referencia.")
                    # Usamos una imagen local genérica real de pulmones
                    st.image("data/generic_xray.jpg", caption=f"Radiografía Genérica (ID Original: {caso.metadata.get('id_caso', '')})", width=400)
                else:
                    st.image(url_img, caption=f"Radiografía ID: {caso.metadata.get('id_caso', '')}", width=400)
            except Exception as e:
                st.error("Error al cargar la imagen.")
        st.caption("🔍 Analiza los datos y escribe tu resolución en el chat.")
    
    # 2. Área de Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Escribe tu diagnóstico y tratamiento propuesto...")
    
    if user_input:
        # Registrar respuesta del estudiante
        st.session_state.messages.append({"role": "human", "content": user_input})
        with st.chat_message("human"):
            st.markdown(user_input)
            
        # Generar Feedback
        with st.chat_message("assistant"):
            with st.spinner("👨‍⚕️ El tutor está evaluando tu respuesta y consultando las normas mexicanas..."):
                # Buscar en ChromaDB guías sobre la enfermedad real del paciente
                enfermedad_real = caso.metadata.get('diagnostico_real', '')
                guias = retriever_guias.invoke(enfermedad_real)
                
                # OPTIMIZACIÓN CRÍTICA: Solo le pasamos a la IA un resumen de 1500 letras de la guía.
                # Si le pasamos las miles de palabras enteras de la guía, la computadora colapsa y tarda 15 minutos en responder.
                texto_guias = guias[0].page_content[:1500] if guias else "Sin guías específicas."
                
                # Obtener calificación del LLM
                feedback = evaluate_user(user_input, caso, texto_guias)
                st.markdown(feedback)
                st.session_state.messages.append({"role": "assistant", "content": feedback})
                
    # 3. Botón para reiniciar
    if len(st.session_state.messages) > 1: # Si el usuario ya interactuó
        st.divider()
        if st.button("Siguiente Paciente ➔"):
            st.session_state.app_state = "inicio"
            st.session_state.current_case = None
            st.rerun()
