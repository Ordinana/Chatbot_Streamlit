__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
debug = True

# Librerías para la preparación de datos
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Librerías para el proceso de Retrieval
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="RAG LANGCHAIN GEMINI App",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",

)

st.title("RAG LANGCHAIN GEMINI App")

file = st.file_uploader("Upload a PDF file", accept_multiple_files=False, type="pdf")
source_data_folder = "./ficheros"
if file:
    with open(source_data_folder+"/pdf.pdf", 'wb') as f: 
        # f.write(file)
        f.write(file.getvalue())

print("librerias cargadas")
# source_data_folder = "./content/MisDatos"
# Leyendo los PDFs del directorio configurado
loader = PyPDFDirectoryLoader(source_data_folder)
data_on_pdf = loader.load()
# cantidad de data cargada
if debug:
    print("pdf cargado")

# Particionando los datos. Con un tamaño delimitado (chunks) y 
# 200 caracters de overlapping para preservar el contexto
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(data_on_pdf)
# Cantidad de chunks obtenidos
print("[INFO] splits preparados")

from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings

# Crea la instancia de embeddings con Cohere
# embeddings_model = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], user_agent="Fran")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                                google_api_key=os.environ["GOOGLE_API_KEY"])

path_db = "./content/VectorDB"  # Ruta a la base de datos del vector store

# Crear el vector store a partir de tus documentos 'splits'
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings_model, 
    persist_directory=path_db
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", 
                             google_api_key=os.environ["GOOGLE_API_KEY"])


retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    # Funcion auxiliar para enviar el contexto al modelo como parte del prompt
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

pregunta = st.text_input("ASK: ", "Que es la pacobilla?")

if st.button("RUN QUERY"):
    response = rag_chain.invoke(pregunta)
    print(response)
    st.write(response)