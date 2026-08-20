import os
import re
import json
import uuid
import traceback
import base64
import hashlib
import secrets
import requests
import asyncio
import threading
import platform
from typing import List, Optional, AsyncGenerator
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import jwt
import uvicorn

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Database Models
from database import (
    SessionLocal, get_db, User, UserSession, UserProgress, AIConversation, AIMessage,
    Question, Quiz, StudentAnswer, Grade, QuestionDifficulty
)

# ---------------------------------------------------------
# App Setup
# ---------------------------------------------------------
load_dotenv()

app = FastAPI(title="IMA Backend", version="2.0.1")

from database import Base, engine

Base.metadata.create_all(bind=engine)
# --- CORS Configuration ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    max_age=3600,
)

# --- Security Config ---
SECRET_KEY = os.getenv("SECRET_KEY", "ima-very-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

security = HTTPBearer(auto_error=False)

# --- GapGPT / AI Clients ---
client = OpenAI(
    api_key=os.getenv("GAPGPT_API_KEY"),
    base_url="https://api.gapgpt.app/v1",
    timeout=30.0,
    max_retries=2
)

async_client = AsyncOpenAI(
    api_key=os.getenv("GAPGPT_API_KEY"),
    base_url="https://api.gapgpt.app/v1",
    timeout=30.0,
    max_retries=2
)

AI_MODEL = "gapgpt-qwen-3.6"
AI_MODEL_FAST = "gapgpt-qwen-3.6"
TTS_MODEL = "tts-1"
# Models for the avatar voice flow (use requested models)
AVATAR_LLM = "gpt-4o-mini"
AVATAR_TTS = "gpt-4o-mini-tts"


# ---------------------------------------------------------
# Security Utils
# ---------------------------------------------------------
def hash_password(password: str) -> str:
    """هش کردن رمز عبور با SHA256 + salt"""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${hashed}"

def verify_password(password: str, hashed: str) -> bool:
    """بررسی و تایید رمز عبور"""
    try:
        salt, original_hash = hashed.split("$")
        new_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return new_hash == original_hash
    except ValueError:
        return False


# ---------------------------------------------------------
# Cache & In-Memory Storage
# ---------------------------------------------------------
class ResponseCache:
    """کش ساده برای بهینه‌سازی پاسخ‌های AI"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get_cache_key(self, messages: list, temperature: float) -> str:
        """ساخت کلید کش بر اساس محتوای پیام و دما"""
        content = json.dumps(messages, sort_keys=True) + str(temperature)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[dict]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: dict):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = {
                "data": value,
                "timestamp": datetime.now(timezone.utc)
            }

chat_cache = ResponseCache(max_size=50)

db_lock = threading.Lock()

# TODO: Replace with actual DB query in production
fake_db = {
    "users": [
        {
            "id": "1",
            "username": "admin",
            "email": "admin@ima.com",
            "full_name": "مدیر سیستم",
            "hashed_password": hash_password("admin123"),
            "avatar": None,
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None,
            "chat_count": 0
        }
    ]
}


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class UserRegister(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    role: Optional[str] = "student"

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    avatar: Optional[str] = None
    role: str
    created_at: str
    chat_count: int

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 800
    stream: bool = False

# --- Teacher Models ---
class QuestionCreate(BaseModel):
    title: str
    description: Optional[str] = None
    content: str
    category: str
    difficulty: str = "medium"
    correct_answer: str
    options: Optional[str] = None
    explanation: Optional[str] = None

class QuestionResponse(BaseModel):
    id: int
    teacher_id: int
    title: str
    description: Optional[str]
    content: str
    category: str
    difficulty: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class QuizCreate(BaseModel):
    title: str
    description: Optional[str] = None
    category: str
    question_ids: List[int]
    time_limit: Optional[int] = None

class QuizResponse(BaseModel):
    id: int
    teacher_id: int
    title: str
    description: Optional[str]
    category: str
    time_limit: Optional[int]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class StudentAnswerCreate(BaseModel):
    question_id: int
    user_answer: str
    is_correct: bool
    score: float = 0

class GradeResponse(BaseModel):
    id: int
    student_id: int
    quiz_id: int
    score: float
    max_score: float
    percentage: float
    submitted_at: datetime
    
    class Config:
        from_attributes = True


# ---------------------------------------------------------
# Core Utilities & Dependencies
# ---------------------------------------------------------
def create_access_token(data: dict) -> str:
    """ساخت توکن JWT"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    """رمزگشایی توکن JWT"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="توکن منقضی شده")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="توکن نامعتبر")

def get_user_by_username(username: str) -> Optional[dict]:
    with db_lock:
        for user in fake_db["users"]:
            if user["username"] == username:
                return user
    return None

def get_user_by_email(email: str) -> Optional[dict]:
    with db_lock:
        for user in fake_db["users"]:
            if user["email"] == email:
                return user
    return None

def user_to_response(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "avatar": user.get("avatar"),
        "role": user.get("role", "user"),
        "created_at": user["created_at"],
        "chat_count": user.get("chat_count", 0)
    }

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security), db: Session = Depends(get_db)):
    """میدلور احراز هویت اجباری کاربر"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="لطفاً وارد حساب کاربری خود شوید",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_token(credentials.credentials)
    username = payload.get("sub")
    
    if not username:
        raise HTTPException(status_code=401, detail="توکن نامعتبر")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="کاربر پیدا نشد")
    
    return user


# ---------------------------------------------------------
# Public Routes & Monitoring
# ---------------------------------------------------------
@app.get("/")
async def root(db: Session = Depends(get_db)):
    users_count = db.query(User).count()
    return {
        "message": "IMA Backend is running",
        "version": "2.0.1",
        "model": AI_MODEL,
        "users_count": users_count
    }

@app.get("/api/ping")
async def ping():
    """بررسی سرعت و دسترس‌پذیری سرور"""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """بررسی سلامت کامل سیستم و ارتباط با هوش مصنوعی"""
    try:
        start = datetime.now()
        client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        latency = (datetime.now() - start).total_seconds()
        users_count = db.query(User).count()
        
        return {
            "status": "healthy",
            "ai_model": AI_MODEL,
            "ai_latency": f"{latency:.2f}s",
            "users_count": users_count,
            "cache_size": len(chat_cache.cache)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


# ---------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------
@app.post("/api/auth/register", response_model=TokenResponse)
async def register(data: UserRegister, db: Session = Depends(get_db)):
    """ثبت نام کاربر جدید با نقش مشخص شده (معلم یا دانش‌آموز)"""
    if len(data.username) < 3:
        raise HTTPException(status_code=400, detail="نام کاربری باید حداقل ۳ کاراکتر باشد")
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="رمز عبور باید حداقل ۶ کاراکتر باشد")
    if "@" not in data.email:
        raise HTTPException(status_code=400, detail="ایمیل نامعتبر است")
    
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="این نام کاربری قبلاً ثبت شده")
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="این ایمیل قبلاً ثبت شده")

    target_role = data.role if data.role in ["student", "teacher", "admin"] else "student"

    new_user = User(
        username=data.username,
        email=data.email,
        full_name=data.full_name,
        password_hash=hash_password(data.password),
        role=target_role
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    token = create_access_token({"sub": new_user.username})
    user_role_str = new_user.role if isinstance(new_user.role, str) else new_user.role.value

    user_dict = {
        "id": str(new_user.id),
        "username": new_user.username,
        "email": new_user.email,
        "full_name": new_user.full_name,
        "avatar": None,
        "role": user_role_str,
        "created_at": new_user.created_at.isoformat(),
        "chat_count": 0
    }
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user_dict)
    }

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(data: UserLogin, db: Session = Depends(get_db)):
    """ورود کاربر به سیستم"""
    user = db.query(User).filter(User.username == data.username).first()
    
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="نام کاربری یا رمز عبور اشتباه است")

    token = create_access_token({"sub": user.username})
    print(f"User logged in: {user.username}")
    
    user_role_str = user.role if isinstance(user.role, str) else user.role.value

    user_dict = {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "avatar": None,
        "role": user_role_str,
        "created_at": user.created_at.isoformat(),
        "chat_count": 0
    }
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user_dict)
    }

@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """دریافت اطلاعات کاربر فعلی"""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        avatar=None,
        role=current_user.role,
        created_at=current_user.created_at.isoformat(),
        chat_count=0
    )


# ---------------------------------------------------------
# Chat & AI Routes
# ---------------------------------------------------------
@app.post("/api/chat")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """چت استاندارد با معلم ریاضی"""
    try:
        messages = []
        for m in request.messages:
            content = m.content[:2000] + "..." if len(m.content) > 2000 else m.content
            messages.append({"role": m.role, "content": content})

        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": "You are IMA (ایما), an expert math teacher assistant. Provide concise, helpful, and direct answers. Get straight to the point without unnecessary pleasantries. Always respond in fluent Persian (Farsi)."
            })

        cache_key = chat_cache.get_cache_key(messages, request.temperature)
        cached_response = chat_cache.get(cache_key)
        
        if cached_response:
            print(f"Cache hit for user: {current_user.username}")
            return cached_response["data"]

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            temperature=min(request.temperature, 0.5),
            max_tokens=min(request.max_tokens, 800),
            presence_penalty=0,
            frequency_penalty=0,
        )

        result = response.model_dump()
        chat_cache.set(cache_key, result)
        
        return result

    except Exception as e:
        print(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """چت استریم برای نمایش لحظه‌ای پاسخ‌ها"""
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": "You are IMA, an expert math teacher. Provide concise and helpful answers. Always respond in fluent Persian (Farsi)."
            })

        async def generate():
            try:
                stream = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=600,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/fast")
async def chat_fast(question: str = Form(...), current_user: User = Depends(get_current_user)):
    """چت سریع - دریافت فقط یک سوال و تولید پاسخ کوتاه"""
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="سوال نمی‌تواند خالی باشد")
        
        question = question[:500]
        messages = [
            {"role": "system", "content": "You are IMA. Provide extremely concise and direct answers without any extra explanation. Always respond in Persian (Farsi)."},
            {"role": "user", "content": question}
        ]

        cache_key = chat_cache.get_cache_key(messages, 0.3)
        cached = chat_cache.get(cache_key)
        if cached:
            return cached["data"]

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            presence_penalty=0,
            frequency_penalty=0,
        )

        result = response.model_dump()
        chat_cache.set(cache_key, result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/vision")
async def chat_vision(
    question: str = Form(""),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user)
):
    """پردازش تصویر و چت (Vision)"""
    try:
        messages = [{
            "role": "system",
            "content": "You are IMA. Analyze the provided image containing a math problem. Solve the problem and provide a short, direct answer. Always respond in fluent Persian (Farsi)."
        }]

        user_content = []

        if file:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="فقط فایل‌های تصویری مجاز هستند")
            
            contents = await file.read()
            
            if len(contents) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="حجم تصویر باید کمتر از 10 مگابایت باشد")
            
            b64 = base64.b64encode(contents).decode('utf-8')
            mime = file.content_type or "image/jpeg"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

        text = question or "لطفاً این سوال ریاضی را حل کن و پاسخ کوتاه بده."
        user_content.append({"type": "text", "text": text[:500]})
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.5,
        )
        
        return response.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/support/chat")
async def support_chat(request: ChatRequest):
    """چت پشتیبانی بدون نیاز به لاگین - مخصوص ویجت سایت"""
    try:
        messages = []
        for m in request.messages:
            content = m.content[:2000] + "..." if len(m.content) > 2000 else m.content
            messages.append({"role": m.role, "content": content})

        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": "You are IMA, a patient and helpful support assistant and math teacher. Provide concise and useful answers. If a question is entirely outside the scope of school mathematics, politely explain that it is outside your area of expertise. Always respond in fluent Persian (Farsi)."
            })

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
            presence_penalty=0,
            frequency_penalty=0,
        )

        return response.model_dump()

    except Exception as e:
        print(f"Support Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# 🤖 AUTHENTICATED DEDICATED AI PLATFORM ROUTES (DATABASE PERSISTENCE)
# =========================================================================
class NewConversationRequest(BaseModel):
    title: str
    model: Optional[str] = AI_MODEL

class NewMessageRequest(BaseModel):
    conversation_id: int
    content: str
    model: Optional[str] = AI_MODEL


def get_recent_conversation_history(db: Session, conversation_id: int, limit: int = 20):
    """فقط آخرین N پیام یک گفتگو را از PostgreSQL می‌خواند و آن‌ها را به ترتیب زمانی قدیمی به جدید برمی‌گرداند."""
    rows = (
        db.query(AIMessage)
        .filter(AIMessage.conversation_id == conversation_id)
        .order_by(AIMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))

@app.get("/api/ai/conversations")
async def get_user_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """دریافت تمام اتاق‌های گفتگوی ذخیره شده دانش‌آموز در دیتابیس"""
    conversations = db.query(AIConversation).filter(
        AIConversation.user_id == current_user.id
    ).order_by(AIConversation.updated_at.desc()).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in conversations]

@app.post("/api/ai/conversations")
async def create_user_conversation(req: NewConversationRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """ایجاد یک نشست گفتگوی جدید در دیتابیس با مدل درخواستی"""
    new_conv = AIConversation(
        user_id=current_user.id,
        title=req.title,
        model=req.model or AI_MODEL
    )
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return {"id": new_conv.id, "title": new_conv.title}

@app.get("/api/ai/conversations/{conv_id}/messages")
async def get_conversation_messages(conv_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """بارگذاری تمام پیام‌های گذشته یک گفتگوی خاص از دیتابیس"""
    conv = db.query(AIConversation).filter(AIConversation.id == conv_id, AIConversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="گفتگو پیدا نشد")
    
    messages = db.query(AIMessage).filter(AIMessage.conversation_id == conv_id).order_by(AIMessage.created_at.asc()).all()
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages]

@app.post("/api/ai/chat")
async def handle_ai_platform_chat(req: NewMessageRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """دریافت پیام، ذخیره در دیتابیس، دریافت پاسخ از هوش مصنوعی با استفاده از مدل پویا و ذخیره پاسخ"""
    conv = db.query(AIConversation).filter(AIConversation.id == req.conversation_id, AIConversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="اتاق گفتگو معتبر نیست")

    # ۱. ثبت پیام دانش‌آموز در دیتابیس
    user_msg = AIMessage(conversation_id=conv.id, role="user", content=req.content)
    db.add(user_msg)
    db.flush()

    # ۲. استخراج پیام‌های اخیر برای حفظ کانتکست کامل گفتگو
    history_logs = get_recent_conversation_history(db, conv.id, 20)

    openai_messages = [{
        "role": "system",
        "content": "You are IMA (ایما), an expert and highly sophisticated math teacher. Help the student solve problems logically, step-by-step. Provide clean equations. Always respond in fluent Persian."
    }]
    for log in history_logs:
        openai_messages.append({"role": log.role, "content": log.content})

    try:
        # ۳. استعلام از سرور هوش مصنوعی با استفاده از مدل دریافت شده
        response = client.chat.completions.create(
            model=req.model or AI_MODEL,
            messages=openai_messages,
            temperature=0.6,
            max_tokens=1000
        )
        ai_reply = response.choices[0].message.content

        # ۴. ثبت پاسخ هوش مصنوعی در دیتابیس
        ai_msg = AIMessage(conversation_id=conv.id, role="assistant", content=ai_reply)
        db.add(ai_msg)
        
        # به‌روزرسانی زمان فعالیت گفتگو
        conv.updated_at = datetime.utcnow()
        db.commit()

        return {"role": "assistant", "content": ai_reply}

    except Exception as e:
        db.rollback()
        print(f"Platform AI Error: {str(e)}")
        raise HTTPException(status_code=500, detail="خطا در پردازش پاسخ هوش مصنوعی")

@app.delete("/api/ai/conversations/{conv_id}")
async def delete_user_conversation(conv_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """حذف کامل یک گفتگو و تمامی پیام‌های آن از دیتابیس"""
    conv = db.query(AIConversation).filter(AIConversation.id == conv_id, AIConversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="آیتم مورد نظر یافت نشد")
    
    db.query(AIMessage).filter(AIMessage.conversation_id == conv_id).delete()
    db.delete(conv)
    db.commit()
    return {"status": "deleted"}

@app.post("/api/ai/vision")
async def handle_ai_platform_vision(
    conversation_id: int = Form(...),
    question: str = Form(""),
    model: str = Form(AI_MODEL),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """پردازش تصویر با مدل انتخابی، دریافت پاسخ و ذخیره کل فرآیند در دیتابیس"""
    conv = db.query(AIConversation).filter(AIConversation.id == conversation_id, AIConversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="اتاق گفتگو معتبر نیست")

    # بررسی حجم و فرمت فایل
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="فقط فایل‌های تصویری مجاز هستند")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="حجم تصویر باید کمتر از 10 مگابایت باشد")

    # ذخیره پیام کاربر در دیتابیس (فقط متن به عنوان رکورد ثبت می‌شود)
    text_log = question if question else "[تصویر ارسال شد]"
    user_msg = AIMessage(conversation_id=conv.id, role="user", content=text_log)
    db.add(user_msg)
    db.flush()

    # تاریخچه اخیر گفتگو را به مدل Vision برای حفظ کانتکست می‌دهیم
    history_logs = get_recent_conversation_history(db, conv.id, 20)

    # تبدیل به Base64 برای OpenAI
    b64 = base64.b64encode(contents).decode('utf-8')
    mime = file.content_type or "image/jpeg"

    messages = [{
        "role": "system",
        "content": "You are IMA, an expert math teacher. Analyze the image, solve the problem clearly using LaTeX for math formulas, and respond in fluent Persian (Farsi)."
    }]

    for log in history_logs:
        if log.role == "user":
            messages.append({"role": "user", "content": log.content})
        else:
            messages.append({"role": "assistant", "content": log.content})

    # ساخت قالب پیام برای مدل Vision
    user_content = [
        {"type": "text", "text": question or "لطفاً این سوال ریاضی را به صورت گام به گام حل کن."[:500]},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    ]
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.5,
        )
        
        ai_reply = response.choices[0].message.content

        # ذخیره پاسخ هوش مصنوعی در دیتابیس
        ai_msg = AIMessage(conversation_id=conv.id, role="assistant", content=ai_reply)
        db.add(ai_msg)
        
        conv.updated_at = datetime.utcnow()
        db.commit()

        return {"role": "assistant", "content": ai_reply}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="خطا در پردازش تصویر توسط هوش مصنوعی")

# =========================================================================
# 🤖 IMA FREE AI — Gemma 4 E2B
# =========================================================================

IMA_GEMMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
IMA_GEMMA_MODEL = "gemma-4-E2B-it"


class IMAFreeChatRequest(BaseModel):
    conversation_id: int
    content: str


@app.post("/api/ai/ima-chat")
async def handle_ima_free_chat(
    req: IMAFreeChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    چت رایگان ایما با Gemma 4 E2B روی سرور اختصاصی.
    """

    # ---------------------------------------------------------
    # 1. بررسی مالکیت گفتگو
    # ---------------------------------------------------------
    conv = db.query(AIConversation).filter(
        AIConversation.id == req.conversation_id,
        AIConversation.user_id == current_user.id
    ).first()

    if not conv:
        raise HTTPException(
            status_code=404,
            detail="گفتگو پیدا نشد"
        )

    if not req.content.strip():
        raise HTTPException(
            status_code=400,
            detail="پیام نمی‌تواند خالی باشد"
        )

    # ---------------------------------------------------------
    # 2. ذخیره پیام کاربر
    # ---------------------------------------------------------
    user_msg = AIMessage(
        conversation_id=conv.id,
        role="user",
        content=req.content
    )

    db.add(user_msg)
    db.flush()

    # ---------------------------------------------------------
    # 3. گرفتن تاریخچه گفتگو (آخرین 20 پیام به ترتیب زمانی)
    # ---------------------------------------------------------
    history_logs = get_recent_conversation_history(db, conv.id, 20)

    messages = [
        {
            "role": "system",
            "content": """
تو «ایما» هستی؛ دستیار هوشمند آموزشی پلتفرم IMA.

قوانین تو:

- همیشه به زبان فارسی روان پاسخ بده.
- پاسخ‌ها را برای دانش‌آموزان، مخصوصاً پایه نهم، قابل فهم ارائه کن.
- در مسائل ریاضی مراحل حل را به صورت منطقی و مرحله‌به‌مرحله توضیح بده.
- فقط جواب نهایی را نده، مگر اینکه کاربر صراحتاً فقط جواب را بخواهد.
- فرمول‌ها را در صورت نیاز با LaTeX بنویس.
- اگر سؤال اشتباه یا ناقص است، محترمانه به کاربر بگو.
- از توضیحات بیش از حد طولانی خودداری کن.
- لحن دوستانه، آموزشی و دقیق داشته باش.
- خودت را ChatGPT معرفی نکن.
- اگر درباره هویتت پرسیده شد، بگو «من ایما، دستیار هوشمند آموزشی IMA هستم.»
"""
        }
    ]

    for log in history_logs:
        messages.append({
            "role": log.role,
            "content": log.content
        })

    # ---------------------------------------------------------
    # 4. درخواست به Gemma (همیشه از مدل ثابت محیطی استفاده می‌شود)
    # ---------------------------------------------------------
    payload = {
        "model": IMA_GEMMA_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 800,
        "reasoning": False
    }

    try:
        response = requests.post(
            IMA_GEMMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()
        choices = result.get("choices") or []
        if not choices or not isinstance(choices, list):
            raise ValueError("Gemma response is missing choices")

        message = choices[0].get("message") or {}
        content = message.get("content")

        if isinstance(content, list):
            ai_reply = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            ).strip()
        else:
            ai_reply = (content or "").strip()

        if not ai_reply:
            raise ValueError("Gemma returned empty response")

        # -----------------------------------------------------
        # 5. ذخیره پاسخ
        # -----------------------------------------------------
        ai_msg = AIMessage(
            conversation_id=conv.id,
            role="assistant",
            content=ai_reply
        )

        db.add(ai_msg)

        conv.updated_at = datetime.utcnow()

        # مشخص می‌کنیم این گفتگو متعلق به ایماست
        conv.model = "gemma-4-E2B-it"

        db.commit()

        return {
            "role": "assistant",
            "content": ai_reply,
            "model": "gemma-4-E2B-it"
        }

    except requests.exceptions.RequestException as e:
        db.rollback()

        print(f"IMA Gemma connection error: {e}")

        raise HTTPException(
            status_code=503,
            detail="سرویس ایما در حال حاضر در دسترس نیست"
        )

    except Exception as e:
        db.rollback()

        print(f"IMA Gemma Error: {e}")

        raise HTTPException(
            status_code=500,
            detail="خطا در پردازش پاسخ ایما"
        )

    
# ---------------------------------------------------------
# Audio Routes (TTS / STT)
# ---------------------------------------------------------
@app.post("/api/tts")
async def tts(
    text: str = Form(""),
    voice: str = Form("alloy"),
    speed: float = Form(1.0),
    current_user: User = Depends(get_current_user)
):
    """تبدیل متن به صوت"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="متن نمی‌تواند خالی باشد")
        
        text = text[:1000] + "..." if len(text) > 1000 else text
        
        fixed = text
        for old, new in [("^2", " به توان دو "), ("^3", " به توان سه "), ("π", " پی ")]:
            fixed = fixed.replace(old, new)

        response = client.audio.speech.create(
            model=TTS_MODEL,
            input=fixed,
            voice=voice,
            speed=float(speed),
            response_format="mp3"
        )

        return Response(
            content=response.content,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
            }
        )

    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stt")
async def stt(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """تبدیل صوت به متن"""
    try:
        allowed_types = ["audio/webm", "audio/mp3", "audio/wav", "audio/mpeg", "audio/ogg"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="فرمت فایل صوتی پشتیبانی نمی‌شود")
        
        audio_bytes = await file.read()
        # choose extension based on content type
        ext = (file.content_type or "audio/webm").split("/")[-1]
        if ext == "mpeg": ext = "mp3"
        if ext not in ["webm", "mp3", "wav", "ogg"]:
            ext = "webm"

        temp_path = f"temp_{uuid.uuid4()}.{ext}"

        try:
            # write bytes to disk
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            # read file and call transcription API (some SDKs expect a file-like object)
            with open(temp_path, "rb") as fh:
                try:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=fh,
                        language="fa"
                    )
                except TypeError:
                    # fallback if SDK expects tuple form
                    fh.seek(0)
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(temp_path, fh, file.content_type or "audio/webm"),
                        language="fa"
                    )

            # attempt to extract text from result in several possible shapes
            text = None
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, dict):
                text = result.get('text') or result.get('transcript')
            else:
                # try to access common attributes
                text = getattr(result, 'data', None)

            return {"text": text or ""}

        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    except Exception as e:
        tb = traceback.format_exc()
        print(f"STT Error: {str(e)}\n{tb}")
        # return safer message to client, but log full traceback server-side
        raise HTTPException(status_code=500, detail="خطا در سرویس تشخیص گفتار")


# Avatar voice endpoint: runs LLM then TTS and returns base64 audio + text
class AvatarRequest(BaseModel):
    text: str
    voice: Optional[str] = "alloy"
    speed: Optional[float] = 1.0


@app.post("/api/avatar/voice_query")
async def avatar_voice_query(req: AvatarRequest, current_user: User = Depends(get_current_user)):
    """دریافت متن، تولید پاسخ با `gpt-4o-mini` و بازگردانی صوت تولیدشده با `gpt-4o-mini-tts` به صورت Base64"""
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="متن نمی‌تواند خالی باشد")

        # 1) LLM reply using the requested avatar model
        messages = [
            {"role": "system", "content": "You are IMA (ایما), an expert Persian math teacher. Answer clearly and concisely in Persian (Farsi)."},
            {"role": "user", "content": req.text}
        ]

        llm_resp = client.chat.completions.create(
            model=AVATAR_LLM,
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        ai_text = llm_resp.choices[0].message.content

        # 2) TTS using the avatar TTS model
        tts_resp = client.audio.speech.create(
            model=AVATAR_TTS,
            input=ai_text,
            voice=req.voice or "alloy",
            speed=float(req.speed or 1.0),
            response_format="mp3"
        )

        audio_b64 = base64.b64encode(tts_resp.content).decode('utf-8')

        return {"text": ai_text, "audio": audio_b64}

    except Exception as e:
        print(f"Avatar voice error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Quiz Generation (Public)
# ---------------------------------------------------------
@app.post("/api/quiz/generate")
async def quiz(
    topic: str = Form(...),
    level: str = Form("medium"),
    count: int = Form(20),
    current_user: User = Depends(get_current_user)
):
    """تولید سوالات کوییز با هوش مصنوعی"""
    if count < 1 or count > 50:
        raise HTTPException(status_code=400, detail="تعداد سوالات باید بین 1 تا 50 باشد")
    
    if level not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="سطح نامعتبر است")

    prompt = f"""Generate {count} multiple-choice math questions about '{topic}' at a '{level}' difficulty level.
Output strictly as a pure JSON array without any markdown formatting or extra text. Use this exact structure:
[
  {{
    "question": "The math question text in Persian",
    "options": ["الف", "ب", "ج", "د"],
    "correct": 0,
    "explanation": "A short explanation of the answer in Persian"
  }}
]
All generated content (questions, options, and explanations) MUST be in fluent Persian (Farsi)."""

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert math quiz generator. Always output pure, valid JSON arrays. Generate content exclusively in Persian (Farsi)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        text = response.choices[0].message.content
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            questions = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            raise HTTPException(status_code=500, detail="خطا در پردازش پاسخ مدل")
        
        if not isinstance(questions, list):
            raise HTTPException(status_code=500, detail="فرمت پاسخ نامعتبر است")
        
        for q in questions:
            if not all(k in q for k in ["question", "options", "correct"]):
                raise HTTPException(status_code=500, detail="ساختار سوالات نامعتبر است")
        
        return {"questions": questions}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="خطا در تولید سوالات. لطفاً دوباره تلاش کنید.")
    except Exception as e:
        print(f"Quiz Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Teacher Panel: Questions & Exams
# ---------------------------------------------------------
@app.post("/api/teacher/questions/create")
async def create_question(
    data: QuestionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """ایجاد دستی سوال جدید توسط معلم"""
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="فقط معلم‌ها می‌توانند سوال ایجاد کنند")
    
    question = Question(
        teacher_id=current_user.id,
        title=data.title,
        description=data.description,
        content=data.content,
        category=data.category,
        difficulty=data.difficulty,
        correct_answer=data.correct_answer,
        options=data.options,
        explanation=data.explanation
    )
    
    db.add(question)
    db.commit()
    db.refresh(question)
    
    return {
        "id": question.id,
        "title": question.title,
        "category": question.category,
        "difficulty": question.difficulty,
        "created_at": question.created_at
    }

@app.get("/api/teacher/questions")
async def get_teacher_questions(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """دریافت تمام سوالات ذخیره‌شده یک معلم"""
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="فقط معلم‌ها می‌توانند سوالات خود را ببینند")
    
    query = db.query(Question).filter(Question.teacher_id == current_user.id)
    if category:
        query = query.filter(Question.category == category)
    
    questions = query.all()
    
    return [
        {
            "id": q.id,
            "title": q.title,
            "category": q.category,
            "difficulty": q.difficulty,
            "created_at": q.created_at
        }
        for q in questions
    ]

@app.get("/api/teacher/questions/{question_id}")
async def get_question_detail(
    question_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """دریافت جزئیات یک سوال خاص"""
    question = db.query(Question).filter(Question.id == question_id).first()
    
    if not question:
        raise HTTPException(status_code=404, detail="سوال پیدا نشد")
    
    if question.teacher_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="دسترسی رد شد")
    
    return {
        "id": question.id,
        "title": question.title,
        "description": question.description,
        "content": question.content,
        "category": question.category,
        "difficulty": question.difficulty,
        "correct_answer": question.correct_answer,
        "options": question.options,
        "explanation": question.explanation,
        "created_at": question.created_at
    }

@app.delete("/api/teacher/questions/{question_id}")
async def delete_question(
    question_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """حذف یک سوال"""
    question = db.query(Question).filter(Question.id == question_id).first()
    
    if not question:
        raise HTTPException(status_code=404, detail="سوال پیدا نشد")
    
    if question.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="فقط معلم مالک می‌تواند سوال را حذف کند")
    
    db.delete(question)
    db.commit()
    
    return {"message": "سوال با موفقیت حذف شد"}

@app.post("/api/teacher/questions/generate")
async def generate_questions_ai(
    category: str = Form(...),
    difficulty: str = Form(...),
    count: int = Form(default=3),
    q_type: str = Form(default="test"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """تولید سوال با هوش مصنوعی و ذخیره خودکار در بانک سوالات"""
    user_role = current_user.role if isinstance(current_user.role, str) else current_user.role.value
    if user_role not in ["teacher", "admin"]:
        raise HTTPException(status_code=403, detail="دسترسی رد شد")
    
    type_instructions = {
        "test": "Multiple-choice questions. Must include 'options' (an array of 4 choices).",
        "blank": "Fill-in-the-blank questions. Include '..........' in the question text where the blank should be. Do not include 'options'.",
        "tf": "True/False questions. Do not include 'options'. The 'correct_answer' must be precisely either 'درست' (True) or 'غلط' (False).",
        "match": "Matching questions. Do not include 'options'.",
        "concept": "Descriptive and conceptual questions. Do not include 'options'.",
        "compute": "Computational and problem-solving questions. Do not include 'options'."
    }
    instruction = type_instructions.get(q_type, type_instructions["test"])
    
    try:
        prompt = f"""Generate {count} math questions.
Topic: '{category}'
Difficulty Level: '{difficulty}'
Question Type: {instruction}

Requirements:
1. Return a pure, valid JSON array of objects.
2. Each object must have the following keys: 'question', 'correct_answer', and 'explanation'.
3. Include the 'options' key ONLY if specified by the Question Type instructions above.
4. Double escape all LaTeX backslashes (e.g., use \\\\frac instead of \\frac).
5. All textual content MUST be written in fluent Persian (Farsi)."""
        
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "You are an advanced math test generator for teachers. Always return pure, valid JSON arrays. Generate all content in Persian (Farsi)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        text = re.sub(r'\\(?![/"\\bfnrt])', r'\\\\', text)
        
        questions = json.loads(text)
        if not isinstance(questions, list):
            questions = [questions]
            
        diff_map = {"easy": QuestionDifficulty.EASY, "medium": QuestionDifficulty.MEDIUM, "hard": QuestionDifficulty.HARD}
        db_difficulty = diff_map.get(difficulty.lower(), QuestionDifficulty.MEDIUM)
        
        for q in questions:
            opts_json = json.dumps(q.get("options")) if q.get("options") else None
            
            new_question = Question(
                teacher_id=current_user.id,
                title=f"سوال هوشمند - {category}",
                content=q.get("question") or q.get("content"),
                category=category,
                difficulty=db_difficulty,
                correct_answer=str(q.get("correct_answer")),
                options=opts_json,
                explanation=q.get("explanation")
            )
            db.add(new_question)
        
        db.commit()
        return {"questions": questions}
        
    except Exception as e:
        print(f"AI Generate Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/questions/global")
async def get_global_question_bank(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """دریافت کل بانک سوالات پلتفرم برای همه معلم‌ها"""
    query = db.query(Question)
    if category:
        query = query.filter(Question.category.ilike(f"%{category}%"))
        
    questions = query.order_by(Question.created_at.desc()).all()
    
    result = []
    for q in questions:
        parsed_opts = None
        if q.options:
            try: 
                parsed_opts = json.loads(q.options)
            except: 
                parsed_opts = [q.options]

        result.append({
            "id": q.id,
            "title": q.title,
            "content": q.content,
            "category": q.category,
            "difficulty": q.difficulty.value if hasattr(q.difficulty, 'value') else str(q.difficulty),
            "options": parsed_opts,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation
        })
    return result

@app.post("/api/teacher/quizzes/create")
async def create_quiz(
    data: QuizCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """ایجاد آزمون جدید"""
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="فقط معلم‌ها می‌توانند آزمون ایجاد کنند")
    
    questions = db.query(Question).filter(
        Question.id.in_(data.question_ids),
        Question.teacher_id == current_user.id
    ).all()
    
    if len(questions) != len(data.question_ids):
        raise HTTPException(status_code=400, detail="برخی از سوالات متعلق به شما نیست")
    
    quiz = Quiz(
        teacher_id=current_user.id,
        title=data.title,
        description=data.description,
        category=data.category,
        question_ids=json.dumps(data.question_ids),
        time_limit=data.time_limit
    )
    
    db.add(quiz)
    db.commit()
    db.refresh(quiz)
    
    return {
        "id": quiz.id,
        "title": quiz.title,
        "category": quiz.category,
        "question_count": len(data.question_ids),
        "created_at": quiz.created_at
    }

@app.post("/api/teacher/quizzes/save")
async def save_quiz_to_db(
    data: QuizCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """ذخیره سبد سوالات به عنوان یک آزمون در دیتابیس"""
    quiz = Quiz(
        teacher_id=current_user.id,
        title=data.title,
        category=data.category,
        question_ids=json.dumps(data.question_ids),
        time_limit=data.time_limit
    )
    db.add(quiz)
    db.commit()
    return {"quiz_id": quiz.id}

@app.get("/api/teacher/quizzes")
async def get_teacher_quizzes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """دریافت تمام آزمون‌های معلم"""
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="فقط معلم‌ها می‌توانند این را ببینند")
    
    quizzes = db.query(Quiz).filter(Quiz.teacher_id == current_user.id).all()
    
    return [
        {
            "id": q.id,
            "title": q.title,
            "category": q.category,
            "question_count": len(json.loads(q.question_ids)) if q.question_ids else 0,
            "is_active": q.is_active,
            "created_at": q.created_at
        }
        for q in quizzes
    ]

@app.get("/api/teacher/quizzes/{quiz_id}/results")
async def get_quiz_results(
    quiz_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """دریافت نتایج یک آزمون خاص"""
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="آزمون پیدا نشد")
    
    if quiz.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="دسترسی رد شد")
    
    grades = db.query(Grade).filter(Grade.quiz_id == quiz_id).all()
    
    return {
        "quiz_id": quiz_id,
        "quiz_title": quiz.title,
        "total_students": len(grades),
        "average_score": sum(g.percentage for g in grades) / len(grades) if grades else 0,
        "results": [
            {
                "student_id": g.student_id,
                "score": g.score,
                "percentage": g.percentage,
                "submitted_at": g.submitted_at
            }
            for g in grades
        ]
    }

@app.get("/api/teacher/quizzes/export-html")
async def export_quiz_html(
    quiz_ids: str,
    include_answers: bool = False,
    db: Session = Depends(get_db)
):
    """تولید قالب HTML استاندارد برای چاپ PDF آزمون"""
    ids = [int(i) for i in quiz_ids.split(",")]
    questions = db.query(Question).filter(Question.id.in_(ids)).all()
    
    html = f"""
    <html lang="fa" dir="rtl">
    <head><style>
        body {{ font-family: 'Vazirmatn', Tahoma, sans-serif; padding: 20mm; }}
        .q-item {{ margin-bottom: 25px; page-break-inside: avoid; }}
        .header {{ text-align: center; border-bottom: 2px solid #000; padding-bottom: 15px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 22px; }}
        .header h2 {{ margin: 5px 0 0 0; font-size: 16px; font-weight: normal; }}
        .student-info {{ display: flex; justify-content: space-between; margin-bottom: 30px; font-weight: bold; font-size: 16px; border: 1px solid #000; padding: 15px; border-radius: 8px; }}
        .answer-key {{ margin-top: 10px; color: #555; border-top: 1px dashed #ccc; padding-top: 10px; }}
        {"" if include_answers else ".answer-key { display: none !important; }"}
    </style></head>
    <body>
        <div class="header">
            <h2>بسمه تعالی</h2>
            <h1>آزمون ارزیابی مستمر ریاضیات</h1>
            <h2>سامانه یکپارچه آموزشی</h2>
        </div>
        <div class="student-info">
            <span>نام و نام خانوادگی: .............................</span>
            <span>کلاس/پایه: .............................</span>
            <span>تاریخ برگزاری: ....../....../......</span>
        </div>
        { "".join([f'<div class="q-item"><h3>سوال {i+1}:</h3><p>{q.content}</p><div class="answer-key">پاسخ: {q.correct_answer}</div></div>' for i, q in enumerate(questions)]) }
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

@app.post("/api/teacher/lesson-plan/generate")
async def generate_lesson_plan_ai(
    topic: str = Form(...),
    grade: str = Form(...),
    time: int = Form(...),
    current_user: User = Depends(get_current_user)
):
    """تولید طرح درس هوشمند با هوش مصنوعی"""
    try:
        prompt = f"""Create a professional math lesson plan for the topic '{topic}', suitable for grade '{grade}', designed for a '{time}'-minute class session.

Output exactly one valid JSON object without any markdown tags. The JSON must contain the following keys:
- "goals": An array of 3 distinct educational goals.
- "intro": A short paragraph to motivate the students.
- "activity": A detailed step-by-step teaching guide and class activities. Use LaTeX for math formulas if needed.
- "evaluation": The method for final assessment.
- "homework": A take-home assignment.

Critical Rules:
1. Double escape all LaTeX backslashes (e.g., use \\\\frac instead of \\frac).
2. All content inside the JSON values MUST be written in professional and educational Persian (Farsi)."""
        
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional math teacher assistant. Return pure, valid JSON only. Generate all content in Persian (Farsi)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        text = response.choices[0].message.content
        text = text.replace("```json", "").replace("```", "").strip()
        text = re.sub(r'\\(?![/"\\bfnrt])', r'\\\\', text)
        
        parsed_plan = json.loads(text)
        return parsed_plan
        
    except Exception as e:
        print(f"Lesson Plan Error: {str(e)}")
        raise HTTPException(status_code=500, detail="خطا در تولید طرح درس")


if __name__ == "__main__":
    import os

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
