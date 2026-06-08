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
    # Encontramos puntos de corte lógicos para separar la Historia Clínica del Manejo/Tratamiento
    keywords_corte = ["tratamiento", "evolución", "evolucion", "se prescribe", "se decide", "intervención", "manejo", "alta", "fallece"]
    texto_lower = texto.lower()
    
    mitad = int(len(texto) * 0.4) 
    corte_idx = len(texto) 
    
    for kw in keywords_corte:
        idx = texto_lower.find(kw, mitad)
        if idx != -1 and idx < corte_idx:
            # Buscamos el final de la oración anterior al corte
            punto_previo = texto.rfind('.', 0, idx)
            salto_previo = texto.rfind('\n', 0, idx)
            mejor_corte = max(punto_previo, salto_previo)
            if mejor_corte != -1 and mejor_corte > mitad:
                corte_idx = mejor_corte + 1
    
    # Si no encuentra ninguna palabra clave clara, corta al 70% como respaldo
    if corte_idx == len(texto):
        corte_idx = int(len(texto) * 0.7)
        ultimo_punto = texto.rfind('.', 0, corte_idx)
        if ultimo_punto != -1:
            corte_idx = ultimo_punto + 1
            
    historia_cruda = texto[:corte_idx].strip()
    diagnostico_y_manejo = texto[corte_idx:].strip()
    
    # Añadimos la nueva leyenda enfocada en manejo clínico
    historia_final = historia_cruda + "\n\n[...El manejo terapéutico y la evolución médica real de este paciente han sido ocultados para que usted proponga su propio abordaje clínico...]"
    
    return historia_final, diagnostico_y_manejo

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
