# Import required libraries
# Import required libraries
from langchain.document_loaders import (  
    PyPDFLoader, # For loading PDF files
    DirectoryLoader, # For loading files from a directory 
    TextLoader, # For loading plain text files
    Docx2txtLoader, # For loading DOCX files
    UnstructuredPowerPointLoader, # For loading PPTX files 
    UnstructuredExcelLoader # For loading XLSX files
)
from langchain.document_loaders.csv_loader import CSVLoader # For loading CSV files  

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv() 

# Set your API keys and environment variables

# OPENAI_API_KEY - OpenAI API key to use their GPT-3 models
openai.api_key = os.getenv('OPENAI_API_KEY')


# Replace with the name of the directory carrying your data
data_directory = "data"

# Load your documents from different sources


def get_documents():

    # Create loaders for PDF, text, CSV, DOCX, PPTX, XLSX files in the specified directory
    pdf_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.csv", loader_cls=CSVLoader)
    docx_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.docx", loader_cls=Docx2txtLoader)
    pptx_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader)
    xlsx_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader)

    # Initialize documents variable
    docs = None

    # Load files using the respective loaders
    pdf_data = pdf_loader.load()
    txt_data = txt_loader.load()
    csv_data = csv_loader.load()
    docx_data = docx_loader.load()
    pptx_data = pptx_loader.load()
    xlsx_data = xlsx_loader.load()

    # Combine all loaded data into a single list
    docs = pdf_data + txt_data + csv_data + docx_data + pptx_data + xlsx_data

    # Return all loaded data
    return docs


# Get the raw documents from different sources
raw_docs = get_documents()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", "\n\n"],
    chunk_size=1500,
    chunk_overlap=20
)

docs = text_splitter.split_documents(raw_docs)

# for doc in docs:
# print(doc)

# Print the number of documents and characters in the first document
print(f'You have {len(docs)} document(s) in your data')
print(
    f'There are {len(docs[0].page_content)} characters in your first document')

# Create OpenAIEmbeddings object using the provided API key
embeddings = OpenAIEmbeddings()

# Create FAISS vector store from the documents and embeddings
db = FAISS.from_documents(docs, embeddings)

# Provide the embeddings instance to associate embeddings with the vector store
try:
    print("Found old vector store, updating it with the new information..")
    old_vectorstore = FAISS.load_local(
        "louisiana_nursery_chatbot_vectorstore", embeddings)
    db.merge_from(old_vectorstore)
    db.save_local("louisiana_nursery_chatbot_vectorstore")
except:
    print(f"No existing vectorstore found. Creating a new one.")
    db.save_local("louisiana_nursery_chatbot_vectorstore")