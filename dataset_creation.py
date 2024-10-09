import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import URI, COLLECTION_NAME, EMBEDDING_MODEL, FILE_PATH
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import jsonlines


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=25)

df = pd.read_csv("news_en_2019.csv")
df = df.loc[(df.text.str.len() > 200) & (df.text.str.len() < 15000)]
docs = text_splitter.create_documents(
    df["text"].tolist(), metadatas=[{"doc_idx": i} for i in range(df.size)]
)

# save to json for bm25 retriever
with jsonlines.open(FILE_PATH, "w") as writer:
    for doc in docs:
        writer.write(doc.model_dump())

# create a vectorstore for vector retrieval
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={"uri": URI},
    collection_name=COLLECTION_NAME,
    drop_old=True,
)
