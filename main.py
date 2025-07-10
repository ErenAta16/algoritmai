from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional
import os
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from services.enhanced_openai_service import enhanced_openai_service
from services.memory_monitor import memory_monitor, monitor_memory
from services.error_handler import error_handler, handle_errors, ErrorSeverity, ErrorCategory
from services.monitoring_service import monitoring_service, monitor_api_call
from services.cache_optimizer import cache_optimizer, query_optimizer
from services.load_balancer import load_balancer, load_balanced, RequestPriority
from services.health_monitor import health_monitor
import re
import html
from collections import defaultdict
import traceback
from logging.handlers import TimedRotatingFileHandler

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# File handler with daily rotation
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = TimedRotatingFileHandler(
    filename=os.path.join(log_dir, "app.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Rate limiting storage
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 30  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit"""
    now = datetime.now()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

# Use enhanced OpenAI service
openai_service = enhanced_openai_service

# Load environment variables
load_dotenv()

# Get allowed origins from environment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,file://").split(",")

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
    allowed_hosts=["localhost", "127.0.0.1", "*.ngrok.io", "*.vercel.app", "*.netlify.app"]
)

# Configure CORS - DEVELOPMENT MODE (allows file:// protocol)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when using wildcard
    allow_methods=["GET", "POST", "OPTIONS"],  # Limit methods
    allow_headers=["Content-Type", "Authorization"],  # Specific headers only
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

# Enhanced Pydantic models with strict validation
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_id: Optional[str] = Field(None, max_length=100, description="Optional user identifier")
    conversation_history: Optional[List] = Field(None, description="Conversation history")

    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        
        # Basic validation only - just strip whitespace
        v = v.strip()
        return v

    @validator('user_id')
    def validate_user_id(cls, v):
        if v is not None:
            # Only allow alphanumeric and basic characters
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError('User ID contains invalid characters')
        return v

    @validator('conversation_history')
    def validate_conversation_history(cls, v):
        if v is not None:
            # Relaxed validation - accept any list
            if not isinstance(v, list):
                v = []
        return v

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
    # Get comprehensive health status
    system_health = health_monitor.get_health_status()
    memory_health = memory_monitor.get_health_status()
    error_health = error_handler.get_health_status()
    load_balancer_health = load_balancer.get_health_status()
    
    # Determine overall status
    overall_status = "healthy"
    if (system_health['overall_status'] in ['critical', 'down'] or 
        memory_health['status'] == 'critical' or 
        error_health['status'] == 'critical' or 
        load_balancer_health['status'] == 'critical'):
        overall_status = "critical"
    elif (system_health['overall_status'] == 'warning' or 
          memory_health['status'] == 'warning' or 
          error_health['status'] == 'warning' or 
          load_balancer_health['status'] == 'warning'):
        overall_status = "warning"
    
    return HealthResponse(
        status=overall_status,
        service="AI Algorithm Consultant Backend",
        timestamp=datetime.now(),
        openai_status="enabled" if openai_service.openai_enabled else "fallback",
        uptime=time.time() - openai_service.last_request_time if hasattr(openai_service, 'last_request_time') else 0
    )

@app.options("/chat")
async def chat_options():
    return {"message": "OK"}

# Advanced monitoring endpoints
@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": monitoring_service.get_metrics_summary(),
            "memory_metrics": memory_monitor.get_memory_stats(),
            "error_metrics": error_handler.get_error_statistics(),
            "load_balancer_metrics": load_balancer.get_load_balancer_stats(),
            "cache_metrics": cache_optimizer.get_stats(),
            "health_status": health_monitor.get_health_status()
        }
    except Exception as e:
        logger.error(f"❌ Metrics endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Metrics temporarily unavailable")

@app.get("/system-status")
async def get_system_status():
    """Get detailed system status"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": health_monitor.get_health_status(),
            "memory_status": memory_monitor.get_health_status(),
            "error_status": error_handler.get_health_status(),
            "load_balancer_status": load_balancer.get_health_status(),
            "monitoring_active": monitoring_service.monitoring_active,
            "auto_recovery_enabled": health_monitor.auto_recovery_enabled
        }
    except Exception as e:
        logger.error(f"❌ System status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="System status temporarily unavailable")

# Background task for analytics
def log_analytics(user_message: str, response_time: float, ai_powered: bool):
    """Log analytics data for monitoring"""
    logger.info(f"📊 Analytics: Message length: {len(user_message)}, Response time: {response_time:.3f}s, AI powered: {ai_powered}")

# AI-powered chat endpoint with advanced monitoring
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, background_tasks: BackgroundTasks, request: Request):
    """
    Professional AI-powered chat endpoint for algorithm consultation - ENHANCED WITH ERROR HANDLING
    """
    start_time = time.time()
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "Unknown")
    
    try:
        # Record API call start
        monitoring_service.increment_counter("api_requests_total")
        monitoring_service.set_gauge("active_connections", len(load_balancer.active_requests))
        
        # Rate limiting check
        if not check_rate_limit(client_ip):
            logger.warning(f"🚫 Rate limit exceeded for IP: {client_ip}")
            monitoring_service.increment_counter("rate_limit_exceeded")
            
            # Log error
            error_handler.log_error(
                error=Exception("Rate limit exceeded"),
                context={"client_ip": client_ip, "endpoint": "/chat"},
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.RATE_LIMIT_ERROR
            )
            
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        # Enhanced input validation
        if not message.message or not message.message.strip():
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
        
        # Get AI response from Enhanced OpenAI service with session management
        try:
            # Extract session ID from headers or generate new one
            session_id = request.headers.get("X-Session-ID")
            
            # If no session ID in header, try to get from conversation history
            if not session_id and message.conversation_history:
                # Look for session ID in conversation history
                for msg in message.conversation_history:
                    if isinstance(msg, dict) and msg.get('session_id'):
                        session_id = msg['session_id']
                        break
            
            result = openai_service.get_chat_response(
                message.message.strip(), 
                message.conversation_history,
                session_id
            )
        except Exception as service_error:
            logger.error(f"❌ OpenAI service error: {str(service_error)}")
            
            # Log error with context
            error_handler.log_error(
                error=service_error,
                context={
                    "client_ip": client_ip,
                    "endpoint": "/chat",
                    "message_length": len(message.message),
                    "user_agent": user_agent
                },
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.API_ERROR
            )
            
            # Record failed API call
            monitoring_service.record_api_call(
                endpoint="/chat",
                method="POST",
                status_code=500,
                response_time=time.time() - start_time,
                user_agent=user_agent,
                ip_address=client_ip
            )
            
            # Return fallback response
            result = {
                "success": False,
                "response": "Üzgünüm, AI servisinde geçici bir sorun yaşıyorum. Yine de size yardımcı olmaya çalışayım!",
                "suggestions": [
                    "Projenizin detaylarını paylaşabilir misiniz?",
                    "Hangi tür veri ile çalışıyorsunuz?",
                    "Projenizin amacı nedir?"
                ],
                "ai_powered": False
            }
        
        processing_time = time.time() - start_time
        
        # Record successful API call
        monitoring_service.record_api_call(
            endpoint="/chat",
            method="POST",
            status_code=200,
            response_time=processing_time,
            user_agent=user_agent,
            ip_address=client_ip
        )
        
        # Record performance metrics
        monitoring_service.record_histogram("response_time", processing_time)
        monitoring_service.increment_counter("successful_requests")
        
        # Add background analytics
        background_tasks.add_task(
            log_analytics, 
            message.message, 
            processing_time, 
            result.get("ai_powered", False)
        )
        
        # Create response
        response = ChatResponse(
            response=result.get("response", "Bir sorun oluştu, lütfen tekrar deneyin."),
            suggestions=result.get("suggestions", []),
            processing_time=processing_time,
            ai_powered=result.get("ai_powered", False)
        )
        
        # Return response with session ID in custom header if available
        if result.get("session_id"):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=response.dict(),
                headers={"X-Session-ID": result["session_id"]}
            )
        
        return response
            
    except HTTPException:
        # Re-raise HTTP exceptions (like rate limiting)
        raise
    except ValidationError as e:
        # Handle Pydantic validation errors
        logger.error(f"❌ Validation error: {str(e)}")
        processing_time = time.time() - start_time
        return ChatResponse(
            response="Mesajınızda bir format hatası var. Lütfen kontrol edip tekrar deneyin.",
            suggestions=[
                "Mesajınızı kontrol edin",
                "Daha basit bir şekilde ifade edin",
                "Tekrar deneyin"
            ],
            processing_time=processing_time,
            ai_powered=False
        )
    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"❌ Unexpected error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        processing_time = time.time() - start_time
        
        # Return professional error response
        return ChatResponse(
            response="Sistem hatası oluştu. Teknik ekibimiz bilgilendirildi. Lütfen daha sonra tekrar deneyin.",
            suggestions=[
                "Daha sonra tekrar deneyin",
                "Farklı bir yaklaşım deneyin",
                "Basit bir soru ile başlayın"
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