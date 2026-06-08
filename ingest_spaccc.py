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
        
        if total_menciones >= 2:
            archivos_resp.append(archivo)

print(f"Se identificaron {len(archivos_resp)} casos relacionados con EPOC.")

# Procesamos TODOS los casos de EPOC encontrados (eliminamos límite de 10)
muestra_archivos = archivos_resp

# 2. Configurar la extracción de la historia clínica (Sin usar IA pesada para ahorrar RAM)
import re

def procesar_texto(texto):
    # Separar en párrafos reales (ignorando saltos de línea vacíos)
    parrafos = [p.strip() for p in texto.split('\n') if len(p.strip()) > 30]
    
    if len(parrafos) <= 2:
        # Si es un texto continuo sin párrafos, cortamos a la mitad en el último punto
        mitad = int(len(texto) * 0.6)
        ultimo_punto = texto.rfind('.', 0, mitad)
        corte = ultimo_punto + 1 if ultimo_punto != -1 else mitad
        historia_cruda = texto[:corte]
        diagnostico = texto[corte:]
    else:
        # Si tiene varios párrafos, los diagnósticos siempre están al final.
        # Nos quedamos con el primer 60% de los párrafos.
        corte_idx = max(1, int(len(parrafos) * 0.6))
        historia_cruda = "\n\n".join(parrafos[:corte_idx])
        diagnostico = "\n\n".join(parrafos[corte_idx:])
        
    # CENSURA DE EMERGENCIA: Si el autor del caso mencionó la enfermedad en la historia, la censuramos.
    # Ej: "Paciente ingresa con diagnóstico de neumonía" -> "Paciente ingresa con diagnóstico de [CENSURADO PARA EXAMEN]"
    historia_limpia = re.sub(r'(?i)(diagnóstico de|compatible con|sugestivo de)\s+([a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+)(,|\.|\n)', r'\1 [CENSURADO PARA EXAMEN]\3', historia_cruda)
    
    historia_final = historia_limpia.strip() + "\n\n[...El diagnóstico final y la evolución médica han sido ocultados para tu evaluación...]"
    
    return historia_final, diagnostico

docs = []
print(f"\nProcesando y curando {len(muestra_archivos)} casos rápidamente...")

for archivo in muestra_archivos:
    with open(os.path.join(corpus_dir, archivo), 'r', encoding='utf-8') as f:
        texto_original = f.read()
    
    try:
        historia, diagnostico = procesar_texto(texto_original)
        
        metadatos = {
            "id_caso": archivo,
            "tipo": "caso_clinico_real",
            "diagnostico_real": diagnostico # Guardamos la resolución completa para el Tutor Socrático
        }
        doc = Document(page_content=historia, metadata=metadatos)
        docs.append(doc)
    except Exception as e:
        print(f"   [!] Error procesando {archivo}: {e}")

print(f"\nSe generaron {len(docs)} expedientes clínicos de alta calidad.")

# 3. Guardar en Base de Datos Vectorial
if len(docs) > 0:
    print("\nCargando Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Guardando en ChromaDB...")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    print("\n¡Ingesta del corpus SciELO completada!")
else:
    print("\nError: No se generó ningún expediente. Revisa los mensajes de error arriba.")
