# Comprobar versiones de librerias para requirements.txt usar "pip freeze" en terminar con env activado

import os
from IPython.display import Markdown

# Librerías para la preparación de datos
from langchain.document_loaders import PyPDFDirectoryLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Librerías para el proceso de Retrieval
from langchain import hub  
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

import streamlit as st
from io import StringIO

import cohere
import numpy as np
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings

import PyPDF2

# Configuración de la página
st.set_page_config(
    page_title="Hazle preguntas a Gemini!",
    page_icon=":rocket:",
    layout="wide",
)

# Encabezados de la interfaz
st.markdown("<h1 style='text-align: center;'>Chatbot de Gemini 🤖</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Sube un archivo PDF, escribe una pregunta y obtén respuestas inteligentes.</h2>", unsafe_allow_html=True)

# Cargar variables de entorno
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Procesamiento del archivo PDF
file = st.file_uploader("Carga un fichero PDF", type="pdf") # Para la subida del pdf
source_data_folder = "./ficheros"

if file:
    with open(source_data_folder + "/PDF.pdf", "wb") as f:
        f.write(file.getvalue())

# Leer el archivo PDF 
loader = PyPDFDirectoryLoader(source_data_folder)
data_on_pdf = loader.load() 

# División del texto. Con un tamaño delimitado (chunks) y 
# 200 caracters de overlapping para preservar el contexto
text_splitter = RecursiveCharacterTextSplitter( 
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(data_on_pdf)
print("-------------------------", splits)

# Crea la instancia de embeddings con Cohere
# embeddings_model = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], user_agent="Ordi")  
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
path_db = "./content/VectorDB"  # Ruta a la base de datos del vector store

# Crear el vector store a partir de tus documentos 'splits'
vectorstore = Chroma.from_documents(   
    documents=splits, 
    embedding=embeddings_model, 
    persist_directory=path_db # persist = guardarlos: Para que podamos persistir los datos y no generarlos todo el tiempo
)

# Definición del modelo LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

# Utiliza el vector de Chroma
retriever = vectorstore.as_retriever()

# Definición del prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    # Funcion auxiliar para enviar el contexto al modelo como parte del prompt
    return "\n\n".join(doc.page_content for doc in docs)

# Define un flujo (chain) 
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt # Es una plantilla donde incluye la  pregunta para el modelo
    | llm # El modelo
    | StrOutputParser() # La forma en la que te devuelve la respuesta del modelo
)

prompt.messages[0].prompt.template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Answer in spanish. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"

# Entrada del usuario y consulta
user_input = st.text_input("Cuál es tu pregunta:")

if st.button("PREGUNTAR", type="primary"):
    if user_input:
        response = rag_chain.invoke(user_input)
        Markdown(response)

        # st.write("Pregunta: " + user_input)
        st.header("Respuesta generada: " )
        st.write(response)