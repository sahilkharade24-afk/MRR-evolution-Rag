import json
import matplotlib.pyplot as plt
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embeddings consistently with retriever.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def calculate_metrics(eval_data, retriever):
    rr_total = 0
    recall_total = 0
    precision_total = 0
    ranks = []

    for item in eval_data:
        docs = retriever.invoke(item["question"])
        rank = None

        for i, doc in enumerate(docs):
            if str(doc.metadata.get("chunk_id")) == str(item["ground_truth_chunk_id"]):
                rank = i + 1
                break

        if rank:
            rr_total += 1 / rank
            recall_total += 1
            precision_total += 1/5 # Simplified P@5: 1 relevant doc found / 5 retrieved
            ranks.append(rank)
        else:
            ranks.append(0)

    count = len(eval_data)
    return {
        "mrr": rr_total / count if count > 0 else 0,
        "recall": recall_total / count if count > 0 else 0,
        "precision": precision_total / count if count > 0 else 0,
        "ranks": ranks
    }

def evaluate_mmr_tuning(generate_chart=False):
    with open("data/eval_data.json") as f:
        eval_data = json.load(f)

    results = []
    # Grid search parameters
    lambda_mults = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    best_mrr = -1
    best_params = {}
    best_ranks = []

    for lm in lambda_mults:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": lm}
        )
        
        metrics = calculate_metrics(eval_data, retriever)
        
        res = {
            "lambda_mult": lm,
            "mrr": metrics["mrr"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "ranks": metrics["ranks"]
        }
        results.append(res)
        
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_params = {"lambda_mult": lm}
            best_ranks = metrics["ranks"]

    return {
        "best_mrr": best_mrr,
        "best_recall": results[results.index(next(r for r in results if r["mrr"] == best_mrr))]["recall"],
        "best_precision": results[results.index(next(r for r in results if r["mrr"] == best_mrr))]["precision"],
        "best_params": best_params,
        "best_ranks": best_ranks,
        "all_results": results
    }

def generate_rank_chart(ranks):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out 0s (not found) and count frequencies of ranks 1-5
    rank_counts = pd.Series([r for r in ranks if r > 0]).value_counts().sort_index()
    
    # Ensure all ranks 1-5 are present in the series
    for i in range(1, 6):
        if i not in rank_counts:
            rank_counts[i] = 0
    rank_counts = rank_counts.sort_index()

    ax.bar(rank_counts.index, rank_counts.values, color='#00ffcc', alpha=0.7)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Retrieved Document Rank Distribution')
    ax.set_xticks(range(1, 6))
    
    # Transparency for glassmorphism embedding
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    return fig

if __name__ == "__main__":
    with open("data/eval_data.json") as f:
        dataset = json.load(f)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("Baseline Metrics:", calculate_metrics(dataset, retriever))
