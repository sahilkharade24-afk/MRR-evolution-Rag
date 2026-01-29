Nfrom langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    print("Loading model...")
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    answer = "The CEO is Raghav Mehta."
    context = ["Raghav Mehta is the CEO.", "The sky is blue."]
    
    print("Embedding answer...")
    ans_emb = model.embed_query(answer)
    print("Embedding context...")
    ctx_embs = model.embed_documents(context)
    
    print("Calculating similarity...")
    sims = cosine_similarity([ans_emb], ctx_embs)[0]
    
    print(f"Similarities: {sims}")
    best_idx = np.argmax(sims)
    print(f"Best match index: {best_idx} (Should be 0)")
    
    if best_idx == 0 and sims[0] > 0.5:
        print("Verification SUCCESS")
    else:
        print("Verification FAILED")
except Exception as e:
    print(f"Error: {e}")
