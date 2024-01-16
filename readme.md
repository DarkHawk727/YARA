# HamiltonBot

This is the repository for the HamiltonBot app. It is a `streamlit` app that utilizes `langchain` and `chromadb` to create a chat-with-pdf/docx-app. This is fairly standard.

```sh
cd .\hamilton_bot\
python -m streamlit run app.py
```

## Roadmap

~~1. Get frontend done using `streamlit`~~
2. Get generic responses from OpenAI
3. Create Tokenization and Embedding for uploaded documents
4. Pass entire document to OpenAI
5. Upload Embeddings to vectorstore
6. Make similarity search through vectorstore and pass into OpenAI
7. Make OpenAI from Azure instead


## File Handling

1. We get `List[UploadedFile]` from `st.file_uploader`. This object has a `.type` attribute that we will have to use with langchain in order to convert it into strings.
2. https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed