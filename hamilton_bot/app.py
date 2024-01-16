from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
import streamlit as st

import tempfile

PDF: str = "application/pdf"
DOCX: str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Setup the DB
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(
        api_key="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",
        openai_api_type="davinci",
    ),
)


st.set_page_config(page_title="HamiltonBot", page_icon="🤖")
st.title("HamiltonBot")
st.text("(Better name pending)")


st.sidebar.title("Files")
files = st.sidebar.file_uploader(
    label="Upload Documents", accept_multiple_files=True, type=["pdf", ".docx"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if files:
    for file in files:
        file_content = file.read()

        # Create a temporary file and write the content
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

            if file.type == "PDF":
                vectordb.add_documents(documents=PyPDFParser(temp_file_path))
            elif file.type == "DOCX":
                vectordb.add_documents(documents=MsWordParser(temp_file_path))

    qa = RetrievalQA.from_llm(
        llm=ChatOpenAI(api_key="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"),
        retriever=vectordb.as_retriever(),
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if message := st.chat_input("Message HamiltonBot..."):
        st.chat_message("user").markdown(message)
        st.session_state.messages.append({"role": "user", "content": message})

        response = qa.invoke({"query": message})

        with st.chat_message("assistant"):
            st.markdown(response["result"])
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    if st.session_state.messages:
        st.session_state.messages = []
    st.error(body="Please upload a pdf/docx/txt file to get started!", icon="🚨")
