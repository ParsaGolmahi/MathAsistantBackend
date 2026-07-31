"""
PostgreSQL Database Configuration and Models
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import enum

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set!")

# برای psycopg2
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )
    
engine = create_engine(
    DATABASE_URL,
    echo=False,  # تغییر به True برای دیدن SQL queries
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ─── Dependency برای FastAPI ───
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- مدل‌های دیتابیس ---

class UserRole(str, enum.Enum):
    """نقش‌های کاربری"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class User(Base):
    """جدول کاربران"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    
    # 🌟 استفاده از String ساده به جای Enum برای هماهنگی آنی
    role = Column(String(20), default="student") 
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

class UserSession(Base):
    """جدول جلسات کاربری"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String(500), unique=True, nullable=False)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id})>"

class UserProgress(Base):
    """جدول پیشرفت کاربر"""
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    course_name = Column(String(100), nullable=False)  # mathlab, mathplanet, etc.
    lessons_completed = Column(Integer, default=0)
    total_lessons = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserProgress(user_id={self.user_id}, course={self.course_name})>"

class AIConversation(Base):
    """جدول گفتگوهای AI"""
    __tablename__ = "ai_conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable برای مهمان‌ها
    session_id = Column(String(100), nullable=True)  # برای مهمان‌های بدون حساب
    title = Column(String(200), nullable=True)
    model = Column(String(50), default="gapgpt-qwen-3.6")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIConversation(id={self.id}, user_id={self.user_id})>"

class AIMessage(Base):
    """جدول پیام‌های گفتگوی AI"""
    __tablename__ = "ai_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("ai_conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" یا "assistant"
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIMessage(id={self.id}, conversation_id={self.conversation_id})>"

class QuestionDifficulty(str, enum.Enum):
    """سطح‌های دشواری سوال"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Question(Base):
    """جدول سوالات ساخته شده توسط معلم"""
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=False)  # محتوای سوال
    category = Column(String(50), nullable=False)  # جبر، هندسه، مثلثات، etc.
    difficulty = Column(Enum(QuestionDifficulty), default=QuestionDifficulty.MEDIUM)
    correct_answer = Column(Text, nullable=False)
    options = Column(Text, nullable=True)  # JSON array گزینه‌های چند‌گزینه‌ای
    explanation = Column(Text, nullable=True)  # توضیح جواب
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Question(id={self.id}, teacher_id={self.teacher_id}, title={self.title})>"

class Quiz(Base):
    """جدول آزمون‌ها"""
    __tablename__ = "quizzes"
    
    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False)
    question_ids = Column(Text, nullable=False)  # JSON array از question_ids
    time_limit = Column(Integer, nullable=True)  # دقیقه
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Quiz(id={self.id}, teacher_id={self.teacher_id}, title={self.title})>"

class StudentAnswer(Base):
    """جدول پاسخ‌های دانش‌آموزان"""
    __tablename__ = "student_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=True)
    user_answer = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    score = Column(Float, default=0)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<StudentAnswer(student_id={self.student_id}, question_id={self.question_id})>"

class Grade(Base):
    """جدول نمرات دانش‌آموزان"""
    __tablename__ = "grades"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False, index=True)
    score = Column(Float, nullable=False)
    max_score = Column(Float, default=100)
    percentage = Column(Float, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Grade(student_id={self.student_id}, quiz_id={self.quiz_id})>"

class AuditLog(Base):
    """جدول ثبت فعالیت‌ها (برای امنیت و پایش)"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AuditLog(action={self.action}, user_id={self.user_id})>"

# --- توابع ایجاد جداول ---

def create_tables():
    """ایجاد تمام جداول"""
    Base.metadata.create_all(bind=engine)
    print("✅ جداول دیتابیس با موفقیت ایجاد شدند!")

def drop_tables():
    """حذف تمام جداول (احتیاط: این عملیات بازگشت‌ناپذیر است)"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️  تمام جداول حذف شدند!")

if __name__ == "__main__":
    print("🔄 در حال بازسازی جداول دیتابیس...")
    drop_tables() 
    create_tables()
    print("✨ جداول با موفقیت بروزرسانی شدند!")
