from rag_chain import create_rag_chain

def start_chat():
    """
    Initializes the RAG chain and starts an interactive chat loop
    that also displays the retrieved context.
    """
    # 1. Create the RAG chain that returns context
    chain = create_rag_chain(return_context=True)
    if not chain:
        print("Failed to initialize RAG chain. Exiting chat.")
        return

    print("\n--- Interactive Chat with Your Document (with Context) ---")
    print('Type "exit" to end the chat.')

    while True:
        try:
            query = input("\nAsk a question: ")
            if query.lower().strip() == "exit":
                print("Exiting chat. Goodbye!")
                break

            # 2. Invoke the chain to get the full response
            response = chain.invoke(query)
            
            # 3. Print the answer
            print("\n--- Answer ---")
            print(response['answer'])
            
            # 4. Print the retrieved context
            print("\n--- Retrieved Context (Sources for the Answer) ---")
            for i, doc in enumerate(response['context']):
                print(f"  Source {i+1} (chunk_id: {doc.metadata.get('chunk_id')}):")
                print(f"    '{doc.page_content[:250]}...'")

        except KeyboardInterrupt:
            print("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    start_chat()