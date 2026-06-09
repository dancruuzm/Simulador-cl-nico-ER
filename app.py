import streamlit as st
import random
import os
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai

st.set_page_config(page_title="Simulador Clínico Enfermedades Respiratorias", page_icon="🩺", layout="wide")

st.title("🩺 Simulador de Casos Clínicos de Enfermedades Respiratorias")
st.markdown("Selecciona tu año de residencia, recibe un caso confirmado de EPOC, propón tu plan de manejo clínico y recibe tutoría socrática especializada.")

@st.cache_resource
def load_rag_system_v5():
    db_path = "./basededatos_final"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    llm = None
    retriever_guias = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"tipo": "documento_teorico"}}
    )
    
    return vectorstore, llm, retriever_guias

with st.spinner("Cargando motor de simulación y guías médicas..."):
    vectorstore, llm, retriever_guias = load_rag_system_v5()

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "simulador" 
if "app_state" not in st.session_state:
    st.session_state.app_state = "inicio" 
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Objetivos de Aprendizaje EPOC ---
OBJETIVOS_EPOC = {
    "R1": "Diagnóstico (FEV1/FVC < 0.70), identificación de errores en espirometría, cálculo del test de Fagerström para dependencia a la nicotina, prescripción de terapia de reemplazo nicotínico, clasificación en paneles ABE, y reconocimiento de exacerbación según criterios ROMA.",
    "R2": "Descarte de diagnósticos alternativos (asma, insuficiencia cardíaca), gasometría y gradiente alveolo-arterial, oxigenoterapia domiciliaria, tratamiento no farmacológico (vacunación/rehabilitación), y terapia broncodilatadora inicial (LAMA/LABA/Corticoides) según eosinófilos.",
    "R3": "Evaluación pronóstica con ÍNDICE BOSEA-90, dosificación de tratamiento médico en exacerbación aguda (corticoides sistémicos 5 días, SABA/SAMA, antibiótico guiado por esputo).",
    "R4": "Criterios para reducción de volumen o trasplante pulmonar, selección de segunda línea tras triple terapia (roflumilast, ensifentrina), uso de biológicos (mepolizumab, dupilumab)."
}

def get_random_case(nivel_residencia):
    filtro_id = f"CASO-{nivel_residencia}.txt"

    try:
        data = vectorstore.get()
        if data and data['documents'] and len(data['documents']) > 0:
            for i, meta in enumerate(data['metadatas']):
                if meta and meta.get("id_caso") == filtro_id:
                    from langchain_core.documents import Document
                    return Document(page_content=data['documents'][i], metadata=meta)
    except Exception as e:
        st.error(f" Error interno al leer ChromaDB: {e}")
    try:
        resultados = vectorstore.similarity_search("paciente clínico", k=100)
        casos_filtrados = [doc for doc in resultados if doc.metadata.get("id_caso") == filtro_id]
        if casos_filtrados:
            return random.choice(casos_filtrados)
    except Exception as e:
        st.error(f" Error en similarity_search: {e}")
        
    return None

def evaluate_user(chat_history, caso_real, contexto_guias, anio_residencia):
    diagnostico_oculto = caso_real.metadata.get('diagnostico_real', 'Desconocido')
    objetivos = OBJETIVOS_EPOC.get(anio_residencia, "")
    
    system_prompt = (
        "=== IDENTIDAD Y ROL ===\n"
        "Actúas como un Tutor Médico Socrático experto. El estudiante YA SABE que el paciente tiene EPOC confirmado. Tu objetivo es guiarlo para que plantee el abordaje clínico y terapéutico correcto, utilizando un máximo de 3 interacciones de guía.\n\n"
        "=== OBJETIVOS DE EVALUACIÓN OBLIGATORIOS ===\n"
        f"El estudiante es un residente de {anio_residencia}. DEBES evaluar su propuesta de tratamiento exigiendo estrictamente que cumpla con los siguientes objetivos:\n"
        f"{objetivos}\n\n"
        "=== REGLAS DE OPERACIÓN ESTRICTAS ===\n"
        "1. Flujo Conversacional: El estudiante propondrá un tratamiento basado en el expediente. Tu deber es analizarlo y guiarlo para que cumpla sus objetivos.\n"
        "2. NUNCA PIDAS MÁS DATOS FÍSICOS DEL PACIENTE: Evalúa al estudiante basándote en los datos que ya están en el expediente.\n"
        "3. Manejo del Método Socrático Puro: NUNCA des la respuesta correcta de inmediato ni confirmes si el diagnóstico o paso es correcto hasta que el estudiante justifique por qué. El estudiante debe ganarse cada pieza de validación.\n"
        "4. Solicitud de Pruebas: Si el estudiante pide una prueba diagnóstica, medicamento o estudio, pregúntale SIEMPRE: '¿Qué esperas encontrar con esta prueba?' o '¿Cuál es la justificación fisiológica para esto?'.\n"
        "5. Manejo de Errores: Si el estudiante solicita un tratamiento contraindicado o inútil, no lo detengas abruptamente. Pregunta: '¿Podrías explicarme la justificación para administrar esto dadas las condiciones actuales del paciente?'. Permite que se den cuenta de su error.\n"
        "6. Retroalimentación Constructiva: Si el alumno argumenta correctamente, felicítalo y pasen al siguiente punto de los objetivos de su año.\n"
        "7. Límite Socrático: Tienes un máximo de 3 interacciones de guía. Haz SOLO UNA PREGUNTA a la vez.\n\n"
        "=== CONDICIONES DE CIERRE (CUÁNDO TERMINAR EL CASO) ===\n"
        "Debes romper el rol socrático y entregar la retroalimentación final integral SÓLO cuando:\n"
        "- Escenario A: El estudiante logra plantear un abordaje que cumple con todos los objetivos de su nivel de residencia.\n"
        "- Escenario B: Se alcanza el límite de 3 interacciones y el estudiante no logra completarlo.\n"
        "- Escenario C: El estudiante se rinde o pide la respuesta directamente.\n\n"
        "=== FORMATO DE SALIDA OBLIGATORIO (JSON) ===\n"
        "Estructura tu respuesta OBLIGATORIAMENTE como JSON:\n"
        "{\n"
        '  "thought": "Evalúo si cumplió los objetivos... Le falta la espirometría... Daré retroalimentación positiva de X y preguntaré por Y.",\n'
        '  "response": "Tu respuesta final (incluyendo la retroalimentación obligatoria + 1 sola pregunta, o la conclusión del caso)."\n'
        "}\n\n"
        "=== CONTEXTO DEL CASO Y GUÍAS (INFORMACIÓN OCULTA PARA EL TUTOR) ===\n"
        f"[RESOLUCIÓN REAL DEL CASO]: {diagnostico_oculto}\n"
        f"[GUÍAS OFICIALES (Úsalas para tu retroalimentación)]: {contexto_guias[:3000]}\n"
    )
    
    import base64
    import httpx
    
    try:
        USER = st.secrets["UNAM_USER"]
        PASSWORD = st.secrets["UNAM_PASSWORD"]
    except KeyError:
        return " Error: No se encontraron las contraseñas en los Secretos de Streamlit."
        
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
        import re
        import json
        
     
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "response" in data:
                    return data["response"]
            except Exception:
                pass
            
        if "[RESPONSE]" in raw_response:
            return raw_response.split("[RESPONSE]")[1].replace("[/RESPONSE]", "").strip()
            
        if "¿" in raw_response and ("We must" in raw_response or "The user" in raw_response or "<|" in raw_response):
            return "¿" + raw_response.split("¿", 1)[1]
     
        return raw_response
    except Exception as e:
        return f"Error al conectar con el tutor remoto: {str(e)}"

def answer_general_query(query, contexto_guias):
    system_prompt = (
        "Eres un experto médico y asistente de biblioteca de la UNAM.\n"
        "Se te ha hecho una pregunta médica general. Usa los siguientes fragmentos de las Guías Clínicas Mexicanas para responder, pero SI LA RESPUESTA NO ESTÁ EN LOS FRAGMENTOS, usa tu conocimiento médico avanzado para responderla de forma completa y correcta.\n\n"
        "=== GUÍAS CLÍNICAS ===\n"
        f"{contexto_guias[:3000]}\n\n"
        "Responde de forma clara, profesional y didáctica."
    )
    
    import base64
    import httpx
    import openai
    try:
        USER = st.secrets["UNAM_USER"]
        PASSWORD = st.secrets["UNAM_PASSWORD"]
    except KeyError:
        return " Error: No se encontraron las contraseñas en los Secretos."
        
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
st.sidebar.title("Modo de Uso")
modo_seleccionado = st.sidebar.radio("Elige una función:", ["Simulador de Casos", "Consulta"])




if modo_seleccionado != st.session_state.app_mode:
    st.session_state.app_mode = modo_seleccionado
    st.rerun()

if st.session_state.app_mode == "Simulador de Casos":
    if st.session_state.app_state == "inicio":
        st.info("👋 Bienvenido al Simulador Clínico de EPOC.")
        nivel_residencia = st.selectbox("Selecciona tu año de residencia:", ["R1", "R2", "R3", "R4"])
        
        if st.button("🩺 Asignarme un Paciente", use_container_width=True):
            caso = get_random_case(nivel_residencia)
            if caso:
                st.session_state.current_case = caso
                st.session_state.residency_year = nivel_residencia
                st.session_state.app_state = "evaluacion"
if nivel_residencia == "R1":
    mensaje_inicial = f"De acuerdo con el expediente clínico proporcionado, analice la información, integre un diagnóstico y proponga su abordaje clínico de acuerdo a sus objetivos de aprendizaje de {nivel_residencia}."
else:
    mensaje_inicial = f"De acuerdo con el expediente clínico de este paciente con diagnóstico confirmado de EPOC. Proponga su abordaje clínico y terapéutico de acuerdo a sus objetivos de aprendizaje de {nivel_residencia}."

st.session_state.messages = [{"role": "assistant", "content": mensaje_inicial}]
    elif st.session_state.app_state == "evaluacion":
        caso = st.session_state.current_case
        
        with st.expander("📄 **Expediente del Paciente (Activo)**", expanded=True):
            st.write(caso.page_content)
            st.caption("🔍 Analiza los datos y escribe tu resolución en el chat.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_input = st.chat_input("Escribe tu plan de abordaje clínico y terapéutico...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
    
            with st.chat_message("assistant"):
                with st.spinner("👨‍⚕️ El tutor está evaluando tu respuesta y consultando las normas mexicanas..."):
                    enfermedad_real = caso.metadata.get('diagnostico_real', '')
                    guias = retriever_guias.invoke(enfermedad_real)
                    
                    texto_guias = guias[0].page_content[:2000] if guias else "Sin guías específicas."
                
                    residency_year = st.session_state.get('residency_year', 'R1')
                    feedback = evaluate_user(st.session_state.messages, caso, texto_guias, residency_year)
                    st.markdown(feedback)
                    st.session_state.messages.append({"role": "assistant", "content": feedback})
        if len(st.session_state.messages) > 1: 
            st.divider()
            if st.button("Siguiente Paciente ➔"):
                st.session_state.app_state = "inicio"
                st.session_state.current_case = None
                st.rerun()

elif st.session_state.app_mode == "Consulta":
    st.info("📚 Bienvenido a la Biblioteca Médica. Hazme cualquier pregunta médica o selecciona una opción rápida.")
    
    st.write("**Preguntas de acceso rápido (Basadas en las nuevas Guías y NOMs):**")
 
    col1, col2, col3 = st.columns(3)
    query = None
    
    if col1.button("Tratamiento Asma", use_container_width=True):
        query = "¿Cuál es el tratamiento farmacológico escalonado para el Asma?"
    if col2.button("Diagnóstico EPOC (2025)", use_container_width=True):
        query = "¿Cuáles son los criterios diagnósticos y tratamiento para EPOC según la guía GMEPOC 2025?"
    if col3.button("Manejo Neumonía", use_container_width=True):
        query = "¿Cuál es el manejo inicial de la Neumonía Adquirida en la Comunidad?"
    col4, col5, col6 = st.columns(3)
    
    if col4.button("Tuberculosis (NOM-006)", use_container_width=True):
        query = "¿Qué establece la NOM-006-SSA2-2013 para la prevención, diagnóstico y tratamiento de la Tuberculosis?"
    if col5.button("Vigilancia Viral", use_container_width=True):
        query = "¿Cuáles son los lineamientos estandarizados para la vigilancia epidemiológica de enfermedades respiratorias virales?"
    if col6.button("Normas Oficiales (SSA)", use_container_width=True):
        query = "¿Cuáles son las normas oficiales mexicanas más importantes para enfermedades respiratorias?"
    
    st.divider()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    query_input = st.chat_input("... O escribe tu duda médica específica (ej. dosis, complicaciones)...")
    
    if query_input:
        query = query_input
    
    if query:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Buscando en las Guías Clínicas..."):
                guias_encontradas = retriever_guias.invoke(query)
                texto_guias_reunidas = "\n\n".join([g.page_content for g in guias_encontradas])
                
                respuesta = answer_general_query(query, texto_guias_reunidas)
            
            st.markdown(respuesta)
            st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})
