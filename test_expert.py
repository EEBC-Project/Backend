
from rag_core import initialize_rag, query_rag
import sys

def test_expert_response():
    print("Initializing RAG...")
    success = initialize_rag()
    if not success:
        print("Failed to initialize RAG")
        return

    question = "Explain the requirements for lighting power density in office buildings."
    print(f"\nAsking Question: {question}")
    
    response = query_rag(question, agent_type="EEBC Expert")
    print("\n--- Response ---\n")
    print(response)
    print("\n----------------\n")

if __name__ == "__main__":
    test_expert_response()
