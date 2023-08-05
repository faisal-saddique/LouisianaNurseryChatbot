import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()

# Create OpenAIEmbeddings object using the provided API key
embeddings = OpenAIEmbeddings()

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your documents')
st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class CustomDataChatbot:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self):

        vectordb =  FAISS.load_local("index\louisiana_nursery_chatbot_vectorstore", embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            qa_chain = self.setup_qa_chain()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()