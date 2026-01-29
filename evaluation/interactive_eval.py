import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from evaluation.recall_precision import recall_at_k, precision_at_k

def interactive_eval():
    print("Initializing Retriever...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Default K=5

    print("\n--- Interactive RAG Evaluation ---")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("\nQ: ").strip()
            if question.lower() == 'exit':
                break
            if not question:
                continue

            print(f"Retrieving top 5 docs...")
            docs = retriever.invoke(question)

            # Display docs for user verification
            print("\n--- Retrieved Chunks ---")
            for i, doc in enumerate(docs):
                chunk_id = doc.metadata.get("chunk_id")
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"Rank {i+1} | ID: {chunk_id} | Content: {content_preview}...")
            print("------------------------")

            # Ask user for Ground Truth
            gt_input = input("Enter correct Chunk ID (or press Enter if Top-1 [Rank 1] is correct): ").strip()
            
            ground_truth_id = None
            if not gt_input and docs:
                 ground_truth_id = docs[0].metadata.get("chunk_id")
                 print(f"(Assuming Rank 1 [ID: {ground_truth_id}] is correct)")
            else:
                 ground_truth_id = gt_input

            # Calculate Metrics
            k = 5
            recall = recall_at_k(docs, ground_truth_id, k)
            precision = precision_at_k(docs, [ground_truth_id], k)

            print("\n--- Results ---")
            print(f"Q: {question}")
            print(f"Recall@{k}: {recall}")
            print(f"Precision@{k}: {precision:.2f}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_eval()
