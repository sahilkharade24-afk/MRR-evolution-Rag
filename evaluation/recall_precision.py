import json
import os
import sys

# Add parent directory to sys.path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def recall_at_k(retrieved_docs, ground_truth_chunk_id, k):
    """
    Recall@K = 1 if correct chunk is found in top-K, else 0
    """
    # Slice to top K
    for doc in retrieved_docs[:k]:
        # Check matching chunk_id
        if str(doc.metadata.get("chunk_id")) == str(ground_truth_chunk_id):
            return 1
    return 0


def precision_at_k(retrieved_docs, relevant_chunk_ids, k):
    """
    Precision@K = relevant chunks in top-K / K
    """
    relevant_count = 0
    # Ensure relevant_chunk_ids are strings for consistent comparison
    relevant_chunk_ids = [str(id) for id in relevant_chunk_ids]
    
    for doc in retrieved_docs[:k]:
        if str(doc.metadata.get("chunk_id")) in relevant_chunk_ids:
            relevant_count += 1
    return relevant_count / k

def load_eval_dataset(path):
    # Construct absolute path if relative is provided
    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), '..'))
        path = os.path.join(base_dir, path)
        
    print(f"Loading dataset from: {path}")
    with open(path, "r") as f:
        return json.load(f)

def evaluate_recall_precision(retriever, k=5, dataset_path="data/eval_data.json"):
    # Ensure path is absolute relative to project root
    if not os.path.isabs(dataset_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dataset_path = os.path.join(base_dir, dataset_path)
    
    dataset = load_eval_dataset(dataset_path)

    recall_scores = []
    precision_scores = []

    print(f"\nStarting Evaluation (k={k})...")

    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth_chunk_id"]

        # Use invoke() instead of get_relevant_documents()
        retrieved_docs = retriever.invoke(question)

        recall = recall_at_k(retrieved_docs, ground_truth, k)
        
        # Assume ground truth is a single item list for precision calculation in this context
        precision = precision_at_k(
            retrieved_docs,
            relevant_chunk_ids=[ground_truth],
            k=k
        )

        recall_scores.append(recall)
        precision_scores.append(precision)

        print(f"Q: {question[:50]}... | Found: {'✅' if recall else '❌'} | P@{k}: {precision:.2f}")

    if not recall_scores:
        print("No data evaluated.")
        return 0, 0

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)

    print("\n📊 FINAL RESULTS")
    print(f"Average Recall@{k}: {avg_recall:.2f}")
    print(f"Average Precision@{k}: {avg_precision:.2f}")

    return avg_recall, avg_precision

if __name__ == "__main__":
    try:
        from retriever import get_retriever
        
        print("Initializing Retriever...")
        # Use a higher fetch_k to ensure we have enough candidates
        retriever = get_retriever(fetch_k=20, lambda_mult=0.5)
        
        if retriever:
            evaluate_recall_precision(retriever, k=5)
        else:
            print("Failed to initialize retriever.")
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure you are running this with the correct python environment.")
