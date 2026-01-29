
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_retriever(fetch_k=10, lambda_mult=0.5):
    """
    Loads the FAISS vector store and creates a retriever with MMR.
    """
    # 1. Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load the FAISS index
    if not os.path.exists("faiss_index"):
        print("Error: FAISS index not found. Please run ingest.py first.")
        return None

    print("Loading FAISS index...")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully.")

    # 3. Create a retriever with MMR
    # For a simple FAISS retriever, MMR is usually configured directly on the retriever
    # If a more complex setup (like ParentDocumentRetriever) is needed, it would involve
    # storing full documents and child chunks separately. For this example, we'll assume
    # the existing FAISS index contains the chunks to be retrieved.
    print("Creating retriever...")
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={'fetch_k': fetch_k, 'lambda_mult': lambda_mult}
    )
    print("Retriever created with MMR search type.")
    return retriever

if __name__ == "__main__":
    retriever = get_retriever()
    if retriever:
        # Example usage:
        query = "What is Apex Dynamics Pvt. Ltd.?"
        print(f"\nRetrieving documents for query: '{query}'")
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content)
