import os
import sys
import requests
import traceback
from dotenv import load_dotenv

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

def check_internet():
    flush_print("Checking internet connectivity...")
    try:
        requests.get("https://www.google.com", timeout=5)
        flush_print("PASS: Internet is reachable.")
        return True
    except Exception as e:
        flush_print(f"FAIL: Internet is not reachable. Error: {e}")
        return False

def check_groq():
    flush_print("\nChecking Groq API...")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        flush_print("FAIL: GROQ_API_KEY not found in .env")
        return False
    
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
        response = llm.invoke("Hello, simple test.")
        flush_print(f"PASS: Groq API responded: {response.content}")
        return True
    except Exception as e:
        flush_print(f"FAIL: Groq API connection failed.")
        traceback.print_exc()
        return False

def check_huggingface():
    flush_print("\nChecking Hugging Face Embeddings...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Try a simple embed
        emb = embeddings.embed_query("test")
        flush_print(f"PASS: Hugging Face Embeddings working (Length: {len(emb)})")
        return True
    except Exception as e:
        flush_print(f"FAIL: Hugging Face Embeddings failed.")
        traceback.print_exc()
        return False

def check_faiss():
    flush_print("\nChecking FAISS Index...")
    if os.path.exists("faiss_index"):
        flush_print("PASS: faiss_index directory exists.")
        return True
    else:
        flush_print("FAIL: faiss_index directory NOT found.")
        return False

if __name__ == "__main__":
    flush_print("Starting Diagnostics...")
    internet = check_internet()
    if internet:
        check_groq()
        check_huggingface()
    check_faiss()
    flush_print("\nDiagnostics Complete.")
