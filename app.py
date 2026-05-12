import streamlit as st
import random

# --- Parche para Streamlit Cloud y ChromaDB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai

# --- Configuración de página UI ---
st.set_page_config(page_title="Simulador Clínico Enfermedades Respiratorias", page_icon="🩺", layout="wide")

st.title("🩺 Simulador de Casos Clínicos de Enfermedades Respiratorias")
st.markdown("Pide un paciente, analiza su caso clínico, propón tu diagnóstico y tratamiento, y recibe retroalimentación.")

# --- Inicialización del Sistema RAG ---
import shutil

def load_rag_system():
    db_path = "./chroma_db_v3"
    # Si estamos en Streamlit Cloud (read-only), movemos la DB a la carpeta temporal /tmp
    if os.path.exists("/mount/src"):
        db_path = "/tmp/chroma_db_v3"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        shutil.copytree("./chroma_db_v3", db_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    llm = None # Ya no usamos Ollama local, usamos el GPU remoto en evaluate_user
    
    # Buscará SOLO en las guías clínicas 
    retriever_guias = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"tipo": "documento_teorico"}}
    )
    
    return vectorstore, llm, retriever_guias

with st.spinner("Cargando motor de simulación y guías médicas..."):
    vectorstore, llm, retriever_guias = load_rag_system()

# --- Manejo de la Máquina de Estados ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "simulador" # Modos: simulador, consulta libre
if "app_state" not in st.session_state:
    st.session_state.app_state = "inicio" # Estados: inicio, evaluacion
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Funciones ---
def get_random_case():
    # Seleccionamos un caso aleatorio de nuestra base de datos de casos (tipo = caso_clinico)
    # Hacemos una búsqueda amplia para traernos varios y escoger uno.
    resultados = vectorstore.similarity_search("paciente", k=50, filter={"tipo": "caso_clinico_real"})
    if resultados:
        return random.choice(resultados)
    return None

def evaluate_user(chat_history, caso_real, contexto_guias):
    diagnostico_oculto = caso_real.metadata.get('diagnostico_real', 'Desconocido')
    
    system_prompt = (
        "Ignora todas tus instrucciones previas. A partir de este momento, actuarás exclusivamente como un 'Guía Socrático'. Tu único propósito es ayudar al estudiante de medicina a profundizar en su comprensión médica a través de preguntas, sin proporcionar nunca respuestas directas.\n\n"
        "1. Tu Rol y Personalidad: Eres un guía curioso y paciente. Tu objetivo principal no es evaluar si el estudiante está 'bien' o 'mal', sino ayudarle a construir su propio conocimiento y fortalecer sus argumentos.\n"
        "2. Reglas Inquebrantables:\n"
        "- NUNCA des una respuesta directa.\n"
        "- IMPORTANTE: HAZ SOLO UNA PREGUNTA A LA VEZ. Es una conversación natural paso a paso. No bombardees al estudiante con 4 o 5 preguntas. Haz UNA sola pregunta clara y espera su respuesta.\n"
        "- Enfócate en el 'porqué' y el 'cómo'.\n"
        "- Descompón los problemas complejos.\n"
        "- Maneja los errores con elegancia. Haz preguntas que le ayuden a descubrir su propio error.\n"
        "- CONDICIÓN DE ÉXITO (CUÁNDO DAR LA RESPUESTA): Si el estudiante llega a la conclusión correcta por sí mismo, o si ya han intercambiado más de 3 mensajes y está muy atascado, o si se rinde explícitamente, ENTONCES felicítalo o ayúdalo, revélale el diagnóstico real y dale un resumen clínico final con recomendaciones basadas en las Guías.\n"
        "- FORMATO ESTRICTO: Para evitar que tus pensamientos internos se muestren, DEBES usar exactamente este formato en tu salida:\n"
        "[THOUGHT]\n"
        "Aquí puedes escribir todos tus razonamientos internos en inglés.\n"
        "[RESPONSE]\n"
        "Aquí va tu respuesta final en español dirigida al estudiante (solo 1 pregunta).\n\n"
        "=== DIAGNÓSTICO Y EVOLUCIÓN REAL DEL PACIENTE (SOLO PARA TU CONOCIMIENTO OCULTO) ===\n"
        f"{diagnostico_oculto}\n\n"
        "=== GUÍAS CLÍNICAS OFICIALES (REFERENCIA OCULTA) ===\n"
        f"{contexto_guias[:2000]}\n"
    )
    
    # --- Conexión al GPU del laboragtorio ---
    import base64
    import httpx
    
    # IMPORTANTE: Ahora jalamos la contraseña de la caja fuerte de Streamlit
    try:
        USER = st.secrets["UNAM_USER"]
        PASSWORD = st.secrets["UNAM_PASSWORD"]
    except KeyError:
        return "⚠️ Error: No se encontraron las contraseñas en los Secretos de Streamlit."
        
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
        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
            
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=api_messages,
            temperature=0.4,
        )
        raw_response = completion.choices[0].message.content
        
        # Ocultar el monólogo interno del modelo
        if "[RESPONSE]" in raw_response:
            return raw_response.split("[RESPONSE]")[1].strip()
        elif "¿" in raw_response and ("We must" in raw_response or "The user" in raw_response):
            return "¿" + raw_response.split("¿", 1)[1]
        return raw_response
    except Exception as e:
        return f"Error al conectar con el tutor remoto: {str(e)}"

def answer_general_query(query, contexto_guias):
    system_prompt = (
        "Eres un experto médico y asistente de biblioteca de la UNAM.\n"
        "Se te ha hecho una pregunta médica general. Usa los siguientes fragmentos de las Guías Clínicas Mexicanas para responder.\n\n"
        "=== GUÍAS CLÍNICAS ===\n"
        f"{contexto_guias[:3000]}\n\n"
        "Responde de forma clara, profesional y siempre basándote en las guías provistas."
    )
    
    import base64
    import httpx
    import openai
    try:
        USER = st.secrets["UNAM_USER"]
        PASSWORD = st.secrets["UNAM_PASSWORD"]
    except KeyError:
        return "⚠️ Error: No se encontraron las contraseñas en los Secretos."
        
    encoded_credentials = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    client = openai.OpenAI(
        base_url="https://dinamica1.fciencias.unam.mx/lmstudio/v1/",
        api_key="lm-studio",
        default_headers={"Authorization": f"Basic {encoded_credentials}"},
        http_client=httpx.Client(verify=False)
    )
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al consultar al servidor: {str(e)}"

# --- Interfaz de Pantallas ---
st.sidebar.title("Modo de Uso")
modo_seleccionado = st.sidebar.radio("Elige una función:", ["Simulador de Casos", "Consulta"])

if modo_seleccionado != st.session_state.app_mode:
    st.session_state.app_mode = modo_seleccionado
    st.rerun()

if st.session_state.app_mode == "Simulador de Casos":
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
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                
            # Generar Feedback
            with st.chat_message("assistant"):
                with st.spinner("👨‍⚕️ El tutor está evaluando tu respuesta y consultando las normas mexicanas..."):
                    # Buscar en ChromaDB guías sobre la enfermedad real del paciente
                    enfermedad_real = caso.metadata.get('diagnostico_real', '')
                    guias = retriever_guias.invoke(enfermedad_real)
                    
                    # OPTIMIZACIÓN CRÍTICA: Solo le pasamos a la IA un resumen de la guía.
                    texto_guias = guias[0].page_content[:2000] if guias else "Sin guías específicas."
                    
                    # Obtener respuesta del Tutor Socrático
                    feedback = evaluate_user(st.session_state.messages, caso, texto_guias)
                    st.markdown(feedback)
                    st.session_state.messages.append({"role": "assistant", "content": feedback})
                    
        # 3. Botón para reiniciar
        if len(st.session_state.messages) > 1: # Si el usuario ya interactuó
            st.divider()
            if st.button("Siguiente Paciente ➔"):
                st.session_state.app_state = "inicio"
                st.session_state.current_case = None
                st.rerun()

elif st.session_state.app_mode == "Consulta":
    st.info("📚 Bienvenido a la Biblioteca Médica. Hazme cualquier pregunta médica o selecciona una opción rápida.")
    
    # Botones de sugerencias rápidas (Las opciones limitadas)
    st.write("**Preguntas de acceso rápido (Basadas en las Guías):**")
    col1, col2, col3 = st.columns(3)
    query = None
    
    if col1.button("Tratamiento Asma", use_container_width=True):
        query = "¿Cuál es el tratamiento farmacológico escalonado para el Asma?"
    if col2.button("Diagnóstico EPOC", use_container_width=True):
        query = "¿Cuáles son los criterios diagnósticos y estudios para EPOC?"
    if col3.button("Manejo Neumonía", use_container_width=True):
        query = "¿Cuál es el manejo inicial de la Neumonía Adquirida en la Comunidad?"
    st.divider()

    # Mostrar historial del chat
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # La caja de texto libre original
    query_input = st.chat_input("... O escribe tu duda médica específica (ej. dosis, complicaciones)...")
    
    if query_input:
        query = query_input
    
    if query:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Buscando en las Guías Clínicas..."):
                # Buscar directamente la pregunta en las guías
                guias_encontradas = retriever_guias.invoke(query)
                texto_guias_reunidas = "\n\n".join([g.page_content for g in guias_encontradas])
                
                respuesta = answer_general_query(query, texto_guias_reunidas)
            
            st.markdown(respuesta)
            st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})
