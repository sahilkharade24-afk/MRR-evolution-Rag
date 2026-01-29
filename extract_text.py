from langchain_community.document_loaders import PyPDFLoader
import os

pdf_path = os.path.join("data", "document", "Sunrise Multispeciality Hospital – Rag Document.pdf")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"Total Pages: {len(pages)}")
for i, page in enumerate(pages[:3]): # Print first 3 pages
    print(f"--- Page {i+1} ---")
    text = page.page_content
    # Print first 500 chars 
    print(text[:500])
    print("-" * 20)
