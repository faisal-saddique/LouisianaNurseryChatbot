# Import required libraries
from langchain.document_loaders import (PyPDFLoader, DirectoryLoader, TextLoader)
from langchain.document_loaders.csv_loader import CSVLoader
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
    # Create loaders for PDF, text, and CSV files in the specified directory
    pdf_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.csv", loader_cls=CSVLoader)

    # Initialize documents variable
    docs = None

    # Load PDF, text, and CSV files using the respective loaders
    pdf_data = pdf_loader.load()  # Load PDF files
    txt_data = txt_loader.load()  # Load text files
    csv_data = csv_loader.load()  # Load CSV files

    # Combine all loaded data into a single list
    docs = pdf_data + txt_data + csv_data

    # Return all loaded data
    return docs

# Get the raw documents from different sources
raw_docs = get_documents()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n","\n\n"],
    chunk_size=1500, 
    chunk_overlap=20
    )

docs = text_splitter.split_documents(raw_docs)

# for doc in docs:
    # print(doc)

# Print the number of documents and characters in the first document
print(f'You have {len(docs)} document(s) in your data')
print(f'There are {len(docs[0].page_content)} characters in your first document')

# Create OpenAIEmbeddings object using the provided API key
embeddings = OpenAIEmbeddings()

# Create FAISS vector store from the documents and embeddings
db = FAISS.from_documents(docs, embeddings)

# Uncomment these lines if you want to store the Vector Database locally for future use.
# You could load it up anytime you want.
# Save the vector database locally in a directory named "faiss_index"
db.save_local("louisiana_nursery_chatbot_vectorstore")

""" 
# Create OpenAI language model
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Create question answering chain using the OpenAI language model
chain = load_qa_chain(llm, chain_type="stuff")

# Start the interactive loop to take user queries
while True:
    query = input("Ask your query:")  # Take user input

    # Perform similarity search in the vector database and get the most similar documents
    docs = db.similarity_search(query, k=3)

    # Run the question answering chain on the selected documents and query
    response = chain.run(input_documents=docs, question=query)

    # Print the response from the question answering chain
    print(f"Response: {response}") """
