# HamiltonBot

This is the repository for the HamiltonBot app. It is a `streamlit` app that utilizes `langchain` and `chromadb` to create a chat-with-pdf/docx-app via OpenAI's GPT series of models. This is fairly standard RAG QnA app. If you want to see just the RAG code, see `RAG-implementation.ipynb`.

## Running the App

You will need to install `python>=3.1.1` and `virtualenv`.

```sh
git clone https://github.com/DarkHawk727/HamiltonBot
cd HamiltonBot
python -m virtualenv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

2. https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed