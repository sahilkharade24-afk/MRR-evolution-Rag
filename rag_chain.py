
import os
from dotenv import load_dotenv
from retriever import get_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(return_context=False):
    """
    Creates a full RAG chain with a retriever and a Groq language model.
    """
    # 1. Load the retriever
    retriever = get_retriever()
    if not retriever:
        print("Failed to get retriever. Exiting.")
        return None

    # 2. Load the Groq API key
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key":
        print("Error: GROQ_API_KEY not found or not set in .env file.")
        print("Please add your Groq API key to the .env file.")
        return None

    # 3. Initialize the Groq Chat model
    print("Initializing Groq Chat model...")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    print("Groq Chat model initialized.")

    # 4. Define the prompt template
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Create the RAG chain
    print("Creating RAG chain...")
    
    # We use a helper to format the context for the prompt
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    if return_context:
        # returns {'answer': ..., 'context': ...}
        from langchain_core.runnables import RunnableParallel
        
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        
        chain = (
            setup_and_retrieval
            | {
                "answer": (lambda x: {"context": format_docs(x["context"]), "question": x["question"]}) | prompt | llm | StrOutputParser(),
                "context": lambda x: x["context"]
            }
        )
    else:
        # returns just the answer string
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
    print("RAG chain created.")
    return chain

if __name__ == "__main__":
    chain = create_rag_chain()
    if chain:
        # Example usage:
        query = "What is the mission statement of Apex Dynamics Pvt. Ltd.?"
        print(f"\n--- Query: {query} ---")
        try:
            response = chain.invoke(query)
            print("\n--- Answer ---")
            print(response)
        except Exception as e:
            print(f"An error occurred while invoking the RAG chain: {e}")
