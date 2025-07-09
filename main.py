from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from services.openai_service import OpenAIService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI service
openai_service = OpenAIService()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Algorithm Consultant API",
    description="Professional Backend API for AI-powered algorithm recommendation system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.ngrok.io"]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm domainlerden erişime izin ver
    allow_credentials=False,  # Credentials false olmalı allow_origins="*" ile
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"🔍 {request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# Pydantic models with validation
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=0, max_length=1000, description="User message")
    user_id: Optional[str] = Field(None, max_length=100, description="Optional user identifier")
    conversation_history: Optional[List[dict]] = Field(None, max_items=50, description="Conversation history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    suggestions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")
    processing_time: Optional[float] = Field(None, description="Response processing time")
    ai_powered: Optional[bool] = Field(None, description="Whether GPT-4 was used")

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    openai_status: str
    uptime: float

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "🤖 AI Algorithm Consultant API is running",
        "version": "2.0.0",
        "status": "healthy"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="AI Algorithm Consultant Backend",
        timestamp=datetime.now(),
        openai_status="enabled" if openai_service.openai_enabled else "fallback",
        uptime=time.time() - openai_service.last_request_time if hasattr(openai_service, 'last_request_time') else 0
    )

@app.options("/chat")
async def chat_options():
    return {"message": "OK"}

# Background task for analytics
def log_analytics(user_message: str, response_time: float, ai_powered: bool):
    """Log analytics data for monitoring"""
    logger.info(f"📊 Analytics: Message length: {len(user_message)}, Response time: {response_time:.3f}s, AI powered: {ai_powered}")

# AI-powered chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, background_tasks: BackgroundTasks):
    """
    Professional AI-powered chat endpoint for algorithm consultation
    """
    start_time = time.time()
    
    try:
        # Handle empty messages gracefully
        if not message.message.strip():
            return ChatResponse(
                response="Merhaba! Size nasıl yardımcı olabilirim? Makine öğrenmesi projeniz hakkında konuşalım!",
                suggestions=[
                    "Makine öğrenmesi projesi yapıyorum",
                    "Hangi algoritma kullanmalıyım?",
                    "Veri analizi yapmak istiyorum",
                    "Yeni başlıyorum, nereden başlamalıyım?"
                ],
                processing_time=time.time() - start_time,
                ai_powered=False
            )
        
        # Get AI response from OpenAI service
        result = openai_service.get_chat_response(
            message.message.strip(), 
            message.conversation_history
        )
        
        processing_time = time.time() - start_time
        
        # Add background analytics
        background_tasks.add_task(
            log_analytics, 
            message.message, 
            processing_time, 
            result.get("ai_powered", False)
        )
        
        if result["success"]:
            return ChatResponse(
                response=result["response"],
                suggestions=result.get("suggestions", []),
                processing_time=processing_time,
                ai_powered=result.get("ai_powered", False)
            )
        else:
            # Return fallback response if OpenAI fails
            return ChatResponse(
                response=result["response"],
                suggestions=result.get("suggestions", []),
                processing_time=processing_time,
                ai_powered=False
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {str(e)}")
        processing_time = time.time() - start_time
        
        # Return professional error response
        return ChatResponse(
            response="Üzgünüm, şu anda teknik bir sorun yaşıyorum. Lütfen daha sonra tekrar deneyin veya sorunuzu farklı şekilde ifade edin.",
            suggestions=[
                "Projenizin detaylarını paylaşabilir misiniz?",
                "Hangi tür veri ile çalışıyorsunuz?",
                "Projenizin amacı nedir?"
            ],
            processing_time=processing_time,
            ai_powered=False
        )

# Algorithm recommendation endpoint
@app.post("/recommend")
async def recommend_algorithms(
    project_type: str = "classification",
    data_size: str = "medium",
    data_type: str = "numerical",
    complexity: str = "medium"
):
    """
    Direct algorithm recommendation endpoint
    """
    try:
        recommendations = openai_service.algorithm_recommender.get_recommendations(
            project_type=project_type,
            data_size=data_size,
            data_type=data_type,
            complexity_preference=complexity
        )
        
        return {
            "recommendations": recommendations,
            "timestamp": datetime.now(),
            "parameters": {
                "project_type": project_type,
                "data_size": data_size,
                "data_type": data_type,
                "complexity": complexity
            }
        }
    except Exception as e:
        logger.error(f"❌ Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Recommendation service temporarily unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5000,
        log_level="info",
        access_log=True
    ) 