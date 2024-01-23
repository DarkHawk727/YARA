from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredAPIFileIOLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_experimental.text_splitter import SemanticChunker


with open("SAMPLE", "r") as f:
    loader = UnstructuredAPIFileIOLoader(
        file=f, mode="elements", show_progress_bar=True,
        api_key="XXXXXXXXXXX",
        post_processors=[clean_extra_whitespace]
    )
docs = loader.load()

embedding = OpenAIEmbeddings(
    api_key="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",
    openai_api_type="davinci",
)

text_splitter = SemanticChunker(embedding=embedding)
chunks = text_splitter.split_documents(docs)


vectordb = Chroma(
    persist_directory="./chroma_db",
).from_documents(documents=chunks, embedding=embedding)

question = "What are the fundamental use cases of superlinear returns?"

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
    llm=ChatOpenAI(
        temperature=0, api_key="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"
    ),
)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
