import jsonlines
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import pandas as pd
from config import URI, COLLECTION_NAME, EMBEDDING_MODEL, RERANKING_MODEL, FILE_PATH
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import ChatCohere
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import click
from rich import print
from prompts import CONTEXTUAL_ANSWER

_ = load_dotenv()

docs = []


with jsonlines.open(FILE_PATH) as reader:
    for obj in reader:
        docs.append(Document(**obj))

df = pd.read_csv("news_en_2019.csv")
llm = ChatCohere(model="command-r-plus")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
model = HuggingFaceCrossEncoder(model_name=EMBEDDING_MODEL)
compressor = CrossEncoderReranker(model=model, top_n=5)

vector_store_loaded = Milvus(
    embeddings,
    connection_args={"uri": URI},
    collection_name=COLLECTION_NAME,
)
retriever = vector_store_loaded.as_retriever(k=10, search_type="similarity")
bm25_retriever = BM25Retriever.from_documents(documents=docs, k=10)

ensemble_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        retriever,
    ],
    weights=[0.5, 0.5],
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)


@click.command()
@click.option("--question", help="Question/Query", required=True)
def get_answer(question):
    print(
        "Warning: The data is from 2018/19, Source: https://github.com/sharad461/english-corpus-nepal"
    )

    # get answer after reranking the documents
    answers = compression_retriever.invoke(question)

    context_lists = []
    chunk_lists = []
    # get the context (document)
    for answer in answers:
        context = df.iloc[answer.metadata["doc_idx"]].text
        context_lists.append(context)
        chunk_lists.append(answer.page_content)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: x["context"], question=lambda x: x["question"],
            chunks=lambda x: x['chunks']
        )
        | CONTEXTUAL_ANSWER
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"context": "".join(context_lists), "question": question,
                           "chunks": "".join(chunk_lists)})
    print(f"Question: {question}\n")
    print("Answers: \n")
    print(answer)


if __name__ == "__main__":
    get_answer()
