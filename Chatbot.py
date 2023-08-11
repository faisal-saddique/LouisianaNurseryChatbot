import utils
import streamlit as st
from streaming import StreamHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Louisiana Nursery", page_icon="https://imgtr.ee/images/2023/08/09/c7a860bf28e9c4c902a73b2e55267477.png")

st.header('Gizmo Chatbot')


# Create OpenAIEmbeddings object using the provided API key
embeddings = OpenAIEmbeddings()


class CustomDataChatbot:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self):

        vectordb = FAISS.load_local(
            "louisiana_nursery_chatbot_vectorstore", embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model,
                         temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            qa_chain = self.setup_qa_chain()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant", avatar="https://imgtr.ee/images/2023/08/09/d63ce87526fd7f310cde6eddcdc9384e.png"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


if __name__ == "__main__":

    obj = CustomDataChatbot()
    obj.main()

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'goodbye'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
