from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredAPIFileIOLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain import Document

from typing import List

import random

FILE_PATH: str = "SAMPLE"
OPENAI_API_KEY: str = "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"
UNSTRUCTURED_API_KEY: str = "XXXXXXXXXXX"
questions: List[str] = [
    "What are the primary goals and objectives for stormwater and drainage management within the City of Hamilton, as outlined in the document?",
    "Can you describe the legislative framework that influences the stormwater management practices in the City of Hamilton?",
    "How does the document address the management of runoff quantity and what specific policies does it propose for flood management and erosion control?",
    "What are the guidelines mentioned for stormwater management in new developments versus existing developments?",
    "How does the City of Hamilton's Storm Drainage Policy approach the management of runoff quality?",
    "Can you explain the role and requirements of the Combined Sewer System as discussed in the document?",
    "What is the Cash-in-Lieu Policy mentioned in the document, and in what context is it applied?",
    "How does the document integrate the Planning and Design Process in stormwater management?",
    "What are the specific challenges and solutions proposed for stormwater management in the context of urban development in Hamilton?",
    "How does the document align with provincial and federal guidelines and objectives in the context of stormwater management?",
]

QUESTION: str = random.choice(questions)

with open(file=FILE_PATH, mode="r") as f:
    loader = UnstructuredAPIFileIOLoader(
        file=f,
        mode="elements",
        show_progress_bar=True,
        api_key=UNSTRUCTURED_API_KEY,
        post_processors=[clean_extra_whitespace],
    )
docs: List[Document] = loader.load()

embedding = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    openai_api_type="davinci",
)

text_splitter = SemanticChunker(embedding=embedding)
chunks: List[str] = text_splitter.split_documents(docs)


vectordb = Chroma(
    persist_directory="./chroma_db",
).from_documents(documents=chunks, embedding=embedding)


retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
    llm=ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY),
)

unique_docs: List[Document] = retriever_from_llm.get_relevant_documents(query=QUESTION)

response = retriever_from_llm.llm.generate_response(
    query=QUESTION, documents=unique_docs, max_tokens=100
)

print(response)
