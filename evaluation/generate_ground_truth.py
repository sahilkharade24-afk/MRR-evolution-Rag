
import json
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def generate_ground_truth():
    eval_path = os.path.join("data", "eval_data.json")
    if not os.path.exists(eval_path):
        print(f"Error: {eval_path} not found")
        return

    print("Loading valid questions...")
    with open(eval_path, "r") as f:
        data = json.load(f)

    # Load resources
    print("Loading retriever...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 1})

    updated_data = []
    print("Generating ground truth...")
    
    for item in data:
        question = item["question"]
        print(f"Processing: {question}")
        
        docs = retriever.invoke(question)
        if docs:
            top_doc = docs[0]
            chunk_id = top_doc.metadata.get("chunk_id")
            item["ground_truth_chunk_id"] = chunk_id
            print(f"  -> Assigned Chunk ID: {chunk_id}")
        else:
            print("  -> No docs found")
            
        updated_data.append(item)

    # Save
    with open(eval_path, "w") as f:
        json.dump(updated_data, f, indent=4)
    print(f"\nUpdated {eval_path} with generated ground truth.")

if __name__ == "__main__":
    generate_ground_truth()
