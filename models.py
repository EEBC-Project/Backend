import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


class Models:
    def __init__(self):
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        
        if groq_api_key:
            try:
                self.model_groq = ChatGroq(
                    api_key=groq_api_key,
                    model=groq_model,
                    temperature=0.1
                )
                print(f"[SUCCESS] Groq model initialized successfully ({groq_model})")
            except Exception as e:
                print(f"[ERROR] Groq initialization failed: {e}")
                raise Exception(f"Groq initialization failed: {e}")
        else:
            print("[ERROR] No GROQ_API_KEY found in environment variables")
            raise Exception("No GROQ_API_KEY found in environment variables")
        
        # Initialize HuggingFace embeddings (free, no API key needed)
        try:
            self.embeddings_hf = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("[SUCCESS] HuggingFace embeddings initialized successfully")
        except Exception as e:
            print(f"[ERROR] HuggingFace embeddings initialization failed: {e}")
            raise Exception(f"HuggingFace embeddings initialization failed: {e}")