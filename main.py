import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.models.schemas import (
    UploadResponse, 
    QuizResponse, 
    QueryRequest, 
    QueryResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.quiz_generator import QuizGenerator

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RAG Quiz Generator",
    description="Generate quizzes from uploaded documents using RAG",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
doc_processor = DocumentProcessor()
quiz_generator = QuizGenerator(
    huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY")
)

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Quiz Generator"}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it for RAG.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process PDF
        document_id, text = doc_processor.process_pdf(file_path)
        
        # Create vector store
        doc_processor.create_vector_store(document_id, text)
        
        return UploadResponse(
            file_name=file.filename,
            document_id=document_id,
            message=f"File {file.filename} uploaded and processed successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(document_id: str, num_questions: int = 5):
    """
    Generate a quiz from the uploaded document.
    """
    if document_id not in doc_processor.document_texts:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        text = doc_processor.document_texts[document_id]
        quiz = quiz_generator.generate_quiz(
            document_id=document_id,
            text=text,
            num_questions=num_questions
        )
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the document using RAG.
    """
    if request.document_id not in doc_processor.vector_stores:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        results = doc_processor.retrieve_context(
            document_id=request.document_id,
            query=request.query,
            num_results=request.num_results
        )
        return QueryResponse(
            query=request.query,
            results=results,
            document_id=request.document_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    List all processed documents.
    """
    return {
        "documents": list(doc_processor.document_texts.keys()),
        "count": len(doc_processor.document_texts)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000))
    )
