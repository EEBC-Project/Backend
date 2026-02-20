
import rag_core
import sys

def verify_prompt():
    print("Verifying EEBC Expert prompt update...")
    
    try:
        prompts = rag_core.AGENT_PROMPTS
        expert_prompt = prompts.get("EEBC Expert", "")
        
        required_phrases = [
            "Senior Technical Expert for the Energy Efficiency Building Code",
            "Explain the 'Why'",
            "Clarify Complexity",
            "Expert Technical Explanation:"
        ]
        
        missing = []
        for phrase in required_phrases:
            if phrase not in expert_prompt:
                missing.append(phrase)
        
        if missing:
            print(f"❌ Verification FAILED. Missing phrases in prompt: {missing}")
            print(f"Current prompt:\n{expert_prompt}")
            sys.exit(1)
        else:
            print("✅ Verification PASSED. EEBC Expert prompt contains all required technical guidelines.")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_prompt()
