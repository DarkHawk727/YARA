import streamlit as st
import time

st.title("HamiltonBot")
st.text("(Better name pending)")


st.sidebar.title("Files")
file = st.sidebar.file_uploader(
    label="Upload a file", accept_multiple_files=True, type=["pdf", ".docx", ".txt"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if file:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = prompt
        with st.spinner("Awaiting response..."):
            # Either leave this in for a bit for "performance improvements" or just remove it lmao
            time.sleep(2)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    if st.session_state.messages:
        st.session_state.messages = []
    # Make it so that if the same file is uploaded, the chat history is saved
    # Or we can just clear it every time a new file is uploaded

    st.error(body="Please upload a pdf/docx/txt file to get started!", icon="🚨")
