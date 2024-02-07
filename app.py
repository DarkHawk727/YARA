from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredFileIOLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from typing import Dict, List

import streamlit as st
import time

OPENAI_API_KEY: str = "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"


st.set_page_config(page_title="HamiltonBot", page_icon="🤖")
st.title("🤖 HamiltonBot")
st.text("(Better name pending)")


st.sidebar.title("Files")

files = st.sidebar.file_uploader(
    label="Upload Documents", accept_multiple_files=True, type=["pdf", ".docx"]
)


st.sidebar.subheader("Previous Conversations")
st.sidebar.selectbox("Select a conversation", ("Conversation 1", "Conversation 2"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if files:
    docs: List[Document] = []
    for file in files:

        loader = UnstructuredFileIOLoader(
            file=file,
            mode="elements",
            strategy="fast",
            post_processors=[clean_extra_whitespace],
        )

        docs.extend(loader.load())

    vectordb = Chroma(
        persist_directory="./chroma_db",
    ).from_documents(
        documents=filter_complex_metadata(documents=docs),
        embedding=OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            openai_api_type="davinci",
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message HamiltonBot..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response: str = qa_chain.invoke(input={"query": prompt})
        with st.chat_message("assistant"):
            st.write(response["result"])
            st.header("Source Documents")
            for doc in response["source_documents"]:
                with st.expander(f"Page #: {doc.metadata['page_number']}"):
                    st.write(doc.page_content)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    if st.session_state.messages:
        st.session_state.messages = []
    st.error(body="Please upload a pdf/docx/txt file to get started!", icon="🚨")
