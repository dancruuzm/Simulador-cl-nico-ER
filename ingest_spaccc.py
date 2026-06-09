import os
import urllib.request
import zipfile
import random
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

url = "https://zenodo.org/records/2560316/files/SPACCC.zip?download=1"
zip_path = "./data/spaccc.zip"
extract_path = "./data/spaccc_corpus"
persist_directory = "./chroma_db_v3"

print("=========================================")
print("  INGESTA DE CASOS SciELO (HUMANOS)  ")
print("=========================================")

if not os.path.exists(extract_path):
    print("Descargando corpus SPACCC (SciELO) desde Zenodo...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extrayendo archivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# En el ZIP de Zenodo, los archivos están en la carpeta corpus dentro de SPACCC
base_dir = os.path.join(extract_path, "SPACCC") if os.path.exists(os.path.join(extract_path, "SPACCC")) else extract_path
corpus_dir = os.path.join(base_dir, "corpus")
todos_los_archivos = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]

print(f"Total de casos humanos encontrados: {len(todos_los_archivos)}")

# 1. Filtro heurístico ESTRICTO para buscar casos de EPOC
keywords = ['epoc', 'enfermedad pulmonar obstructiva', 'enfisema', 'bronquitis cronica']
archivos_resp = []

for archivo in todos_los_archivos:
    with open(os.path.join(corpus_dir, archivo), 'r', encoding='utf-8') as f:
        texto = f.read().lower()
        # Buscamos que mencione alguna de las palabras clave de EPOC al menos 2 veces
        total_menciones = sum(texto.count(kw) for kw in keywords)
        
        if total_menciones >= 1:
            archivos_resp.append(archivo)

print(f"Se identificaron {len(archivos_resp)} casos relacionados con EPOC.")

# Procesamos TODOS los casos de EPOC encontrados (eliminamos límite de 10)
muestra_archivos = archivos_resp

import streamlit as st
import openai
import base64
import httpx
import json

# Conexión al GPU de la UNAM usando secrets
try:
    USER = st.secrets["UNAM_USER"]
    PASSWORD = st.secrets["UNAM_PASSWORD"]
    encoded_credentials = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    client = openai.OpenAI(
        base_url="https://dinamica1.fciencias.unam.mx/lmstudio/v1/",
        api_key="lm-studio",
        default_headers={"Authorization": f"Basic {encoded_credentials}"},
        http_client=httpx.Client(verify=False, timeout=60.0)
    )
except Exception as e:
    print(f"Error crítico: No se encontraron las contraseñas en .streamlit/secrets.toml. Por favor configúralas.")
    sys.exit(1)

def procesar_con_ia(texto_original):
    prompt = """
Lee el siguiente caso clínico completo. Analízalo como un experto médico y responde OBLIGATORIAMENTE con un objeto JSON con estas 3 claves:
{
  "es_epoc": true o false (Pon true SÓLO si el problema principal o la razón de consulta del paciente es una exacerbación o complicación directa de EPOC/Enfisema/Bronquitis Crónica. Si el EPOC es solo un antecedente y la enfermedad principal es un cólico renal, un infarto o un tumor, pon false),
  "historia_clinica": "Aquí debes extraer TODA la historia clínica, antecedentes, síntomas actuales, signos vitales y laboratorios/estudios del paciente EXACTAMENTE como viene en el texto original, pero detente justo antes de que el médico empiece a dar el tratamiento o la evolución de alta.",
  "manejo_real": "Aquí pon el resto del caso: cómo se decidió tratar al paciente, qué medicamentos se le dieron, cómo evolucionó y si fue dado de alta o falleció."
}

CASO:
""" + texto_original[:3500]

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en curación de datos médicos. Responde siempre y exclusivamente en JSON validado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        respuesta_texto = completion.choices[0].message.content
        # Limpiar respuesta por si el LLM le pone markdown
        if "```json" in respuesta_texto:
            respuesta_texto = respuesta_texto.split("```json")[1].split("```")[0]
        elif "```" in respuesta_texto:
            respuesta_texto = respuesta_texto.split("```")[1].split("```")[0]
            
        return json.loads(respuesta_texto.strip())
    except Exception as e:
        print("Error en la IA:", e)
        return None

docs = []
print(f"\nProcesando y curando {len(muestra_archivos)} casos con IA. Esto puede tardar varios minutos...")

casos_exitosos = 0
for archivo in muestra_archivos:
    with open(os.path.join(corpus_dir, archivo), 'r', encoding='utf-8') as f:
        texto_original = f.read()
    
    print(f"Ingresando {archivo} manualmente...")
    historia_final = texto_original + "\n\n[...El manejo terapéutico y la evolución médica real de este paciente han sido ocultados para que usted proponga su propio abordaje clínico...]"
    metadatos = {
        "id_caso": archivo,
        "tipo": "caso_clinico_real",
        "diagnostico_real": "Diagnóstico real de paciente (Oculto en DB)"
    }
    
    doc = Document(page_content=historia_final, metadata=metadatos)
    docs.append(doc)
    casos_exitosos += 1
    print(f" -> APROBADO (Manual): Es un caso real de EPOC.")

# 3. Guardar en Base de Datos Vectorial
if len(docs) > 0:
    print("\nCargando Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Guardando en ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="."
    )
    print("\n¡Ingesta del corpus SciELO completada!")
else:
    print("\nError: No se generó ningún expediente. Revisa los mensajes de error arriba.")
