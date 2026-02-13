import os
from dotenv import load_dotenv
from rag_core import initialize_rag, query_rag
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from models import Models

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
    agent_type: str = "EEBC Expert"

# Response models
class RAGResponse(BaseModel):
    question: str
    answer: str
    timestamp: str
    status: str = "success"

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    chunks_created: int

# -------- REST API Endpoints -------- #

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Service API",
        "status": "running",
        "endpoints": {
            "upload": "POST /upload",
            "rag": "POST /tools/rag",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG Service"}

# PDF Upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and ingest it into the RAG system"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("./Uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[INFO] File saved: {file_path}")
        
        # Initialize models for ingestion
        models = Models()
        embeddings = models.embeddings_hf
        
        # Initialize vector store
        vector_store = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory="./DB/chroma_langchain_db"
        )
        
        # Load and process the PDF
        print(f"[INFO] Loading document: {file_path}")
        loader = PyPDFLoader(str(file_path))
        loaded_documents = loader.load()
        
        if not loaded_documents:
            raise HTTPException(status_code=400, detail="No content found in PDF")
        
        print(f"[INFO] Splitting {len(loaded_documents)} pages into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(loaded_documents)
        
        print(f"[INFO] Created {len(documents)} chunks")
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Add documents to vector store
        print("[INFO] Adding documents to vector store...")
        vector_store.add_documents(documents=documents, ids=uuids)
        print("[SUCCESS] Ingestion complete")
        
        # Note: No need to reinitialize RAG system - the retriever will automatically
        # query the updated vector store on the next request
        
        return UploadResponse(
            filename=file.filename,
            status="success",
            message="PDF uploaded and processed successfully",
            chunks_created=len(documents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# RAG endpoint
@app.post("/tools/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    """Ask a question to the RAG system"""
    try:
        # Get the answer from the RAG system
        answer = query_rag(request.question, request.agent_type)
        
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
    print(f"🔑 Groq API Key: {'✅ Configured' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print(f"🤖 Model: {os.getenv('GROQ_MODEL', 'llama3-70b-8192')}")
    
    # Configure server settings
    http_port = int(os.getenv('PORT', 8000))     # Default to 8000 for HTTP
    
    print(f"🌐 Server will be accessible at:")
    print(f"   • HTTP API: http://localhost:{http_port}")
    print(f"   • HTTP API Docs: http://localhost:{http_port}/docs")
    print(f"   • Upload PDF: POST http://localhost:{http_port}/upload")
    print()
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=http_port)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
