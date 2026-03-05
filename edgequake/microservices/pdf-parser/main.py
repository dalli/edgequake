import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Edgequake PDF Parser API", version="1.0.0")

class ParseResponse(BaseModel):
    markdown: str
    pages: int
    status: str
    error: str | None = None

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "pdf-parser"}

@app.post("/api/v1/parse", response_model=ParseResponse)
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save the uploaded file to a temporary location
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Call parser.py to process the PDF
        from parser import process_pdf
        markdown, pages, elements = process_pdf(temp_path)
        
        return ParseResponse(
            markdown=markdown,
            pages=pages,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
