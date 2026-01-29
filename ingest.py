from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json

with open("data/documents/apex_dynamics.txt") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

docs = []
for i, chunk in enumerate(chunks):
    docs.append(
        Document(
            page_content=chunk,
            metadata={"chunk_id": f"chunk_{i}"}
        )
    )

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vectorstore.save_local("faiss_index")
