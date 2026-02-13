from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from models import Models

# Initialize The Models (OpenAI only)
models = Models()

# Use OpenAI embeddings and chat model
embeddings = models.embeddings_hf
llm = models.model_groq
print("[SUCCESS] Using HuggingFace embeddings and Groq chat model")

# Initialize the vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./DB/chroma_langchain_db"
)

# Define the chat prompt
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
)

#Define the retrieval chain
retriever = vector_store.as_retriever(search_kwargs={"k":10})

# Create the retrieval chain using LCEL
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("[SUCCESS] Chat system ready!")

# Main Loop

def main():
    import sys
    
    # Check if question is provided as command line argument
    if len(sys.argv) > 1:
        # Non-interactive mode
        query = " ".join(sys.argv[1:])
        try:
            response = retrieval_chain.invoke(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Interactive mode
    print("\n=== RAG Service Chat Interface ===")
    print("Ask questions about the ingested documents. Type 'q', 'quit' or 'exit' to end.")
    print()
    
    while True:
        query = input("User: ")
        if query.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
            
        try:
            print("🔍 Searching documents...")
            response = retrieval_chain.invoke(query)
            print("Assistant:", response)
            print()
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            print("Please try rephrasing your question.")
            print()

if __name__ == "__main__":
    main()