import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

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
        
        # Initialize HuggingFace embeddings via API (lightweight, no local model)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("[ERROR] HF_TOKEN missing. Required for lightweight embeddings.")
            raise Exception("HF_TOKEN missing in environment variables")

        try:
            # Use HuggingFaceEndpointEmbeddings which replaces the old Inference API class
            self.embeddings_hf = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                task="feature-extraction",
                huggingfacehub_api_token=hf_token
            )
            print("[SUCCESS] HuggingFace Inference API embeddings initialized successfully")
        except Exception as e:
            print(f"[ERROR] HuggingFace API initialization failed: {e}")
            raise Exception(f"HuggingFace API initialization failed: {e}")