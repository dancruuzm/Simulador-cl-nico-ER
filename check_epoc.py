import os
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma(persist_directory='./chroma_db_v3', embedding_function=embeddings)

results = vectorstore.similarity_search("EPOC enfermedad pulmonar obstructiva cronica", k=100, filter={"tipo": "caso_clinico_real"})

seen = set()
unique_cases = []
for r in results:
    case_id = r.metadata.get('id_caso')
    if case_id and case_id not in seen:
        seen.add(case_id)
        unique_cases.append(case_id)

print(f"Encontrados {len(unique_cases)} casos únicos:")
for c in unique_cases:
    print(f"- {c}")
