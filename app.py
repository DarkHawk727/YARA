import streamlit as st
import time

st.set_page_config(page_title="HamiltonBot", page_icon="🤖")
st.title("🤖 HamiltonBot")
st.text("(Better name pending)")


st.sidebar.title("Files")

files = st.sidebar.file_uploader(label="Upload Documents", accept_multiple_files=True, type=["pdf", ".docx"])


st.sidebar.subheader("Previous Conversations")
st.sidebar.selectbox("Select a conversation", ("Conversation 1", "Conversation 2"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if files:

    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message HamiltonBot..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response: str = f">\"CITATION\"\n\n {prompt}"
        with st.status("Awaiting Response..."):
            st.write("Creating Multiple Queries...")
            time.sleep(2)
            st.write("Retrieving Chunks...")
            time.sleep(1)
            st.write("Sending to LLM...")
            time.sleep(1)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    if st.session_state.messages:
        st.session_state.messages = []
    st.error(body="Please upload a pdf/docx/txt file to get started!", icon="🚨")
