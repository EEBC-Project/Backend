import os
from dotenv import load_dotenv
from rag_core import initialize_rag, query_rag
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service API",
    description="REST API for RAG-based CV Question Answering",
    version="1.0.0"
)

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")
if len(origins) == 1 and origins[0] == "*":
    # Allow all origins mode
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    print(f"[CORS] Allowing requests from all origins")
else:
    # Restricted origins mode
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    print(f"[CORS] Allowing requests from: {', '.join(origins)}")

# Request models
class RAGRequest(BaseModel):
    question: str

# Response models
class RAGResponse(BaseModel):
    question: str
    answer: str
    timestamp: str
    status: str = "success"

# -------- REST API Endpoints -------- #

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Service API",
        "status": "running",
        "endpoints": {
            "rag": "POST /tools/rag",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG Service"}

# RAG endpoint
@app.post("/tools/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    """Ask a question to the RAG system"""
    try:
        # Get the answer from the RAG system
        answer = query_rag(request.question)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return RAGResponse(
            question=request.question, 
            answer=answer,
            timestamp=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# -------- Run RAG Service -------- #
if __name__ == "__main__":
    import logging
    import sys
    
    # Load environment variables
    load_dotenv()
    
    # Initialize RAG system at startup
    print("🔄 Initializing RAG system...")
    try:
        initialize_rag()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        sys.exit(1)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting RAG Service...")
    print(f"📍 Environment: {'Production' if os.getenv('RENDER') else 'Development'}")
    print(f"🔑 OpenAI API Key: {'✅ Configured' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    
    # Configure server settings
    http_port = int(os.getenv('PORT', 8000))     # Default to 8000 for HTTP
    
    print(f"🌐 Server will be accessible at:")
    print(f"   • HTTP API: http://localhost:{http_port}")
    print(f"   • HTTP API Docs: http://localhost:{http_port}/docs")
    print()
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=http_port)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
