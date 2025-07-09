from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from services.openai_service import OpenAIService

# Initialize OpenAI service
openai_service = OpenAIService()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Algorithm Consultant API",
    description="Backend API for AI-powered algorithm recommendation system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm domainlerden erişime izin ver
    allow_credentials=False,  # Credentials false olmalı allow_origins="*" ile
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = None

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "AI Algorithm Consultant API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Algorithm Consultant Backend"}

@app.options("/chat")
async def chat_options():
    return {"message": "OK"}

# AI-powered chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """
    AI-powered chat endpoint for algorithm consultation
    """
    try:
        # Get AI response from OpenAI service
        result = openai_service.get_chat_response(
            message.message, 
            message.conversation_history
        )
        
        if result["success"]:
            return ChatResponse(
                response=result["response"],
                suggestions=result["suggestions"]
            )
        else:
            # Return fallback response if OpenAI fails
            return ChatResponse(
                response=result["response"],
                suggestions=result["suggestions"]
            )
            
    except Exception as e:
        # Fallback response for any unexpected errors
        return ChatResponse(
            response="Üzgünüm, şu anda teknik bir sorun yaşıyorum. Lütfen daha sonra tekrar deneyin.",
            suggestions=[
                "Projenizin detaylarını paylaşabilir misiniz?",
                "Hangi tür veri ile çalışıyorsunuz?",
                "Projenizin amacı nedir?"
            ]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 