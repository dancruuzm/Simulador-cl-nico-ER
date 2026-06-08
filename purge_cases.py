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

try:
    # Delete all documents that are cases
    vectorstore._collection.delete(where={"tipo": "caso_clinico_real"})
    print("Casos antiguos eliminados de la base de datos.")
except Exception as e:
    print(f"Error o no habia casos: {e}")
