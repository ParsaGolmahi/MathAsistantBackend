"""
این سکریپت جداول دیتابیس را ایجاد می‌کند
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

# تنظیمات دیتابیس
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ima_db")

def connect_to_postgresql():
    """اتصال به سرور PostgreSQL بدون انتخاب دیتابیس"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database="postgres"  # دیتابیس پیش‌فرض
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except psycopg2.Error as e:
        print(f"❌ خطا در اتصال به PostgreSQL: {e}")
        return None

def create_database():
    """ایجاد دیتابیس جدید"""
    conn = connect_to_postgresql()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # بررسی وجود دیتابیس
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_NAME,)
        )
        
        if cursor.fetchone():
            print(f"ℹ️  دیتابیس '{DB_NAME}' قبلاً وجود دارد.")
        else:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"✅ دیتابیس '{DB_NAME}' با موفقیت ایجاد شد!")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ خطا در ایجاد دیتابیس: {e}")
        return False

def create_tables():
    """ایجاد جداول با استفاده از SQLAlchemy"""
    try:
        from database import create_tables
        create_tables()
        print("✅ تمام جداول با موفقیت ایجاد شدند!")
        return True
    except Exception as e:
        print(f"❌ خطا در ایجاد جداول: {e}")
        return False

def setup_database():
    """راه‌اندازی کامل دیتابیس"""
    print("=" * 50)
    print("🚀 شروع راه‌اندازی دیتابیس...")
    print("=" * 50)
    
    print(f"\n📋 تنظیمات اتصال:")
    print(f"   Host: {DB_HOST}")
    print(f"   Port: {DB_PORT}")
    print(f"   User: {DB_USER}")
    print(f"   Database: {DB_NAME}")
    
    # گام 1: ایجاد دیتابیس
    print("\n1️⃣  ایجاد دیتابیس...")
    if not create_database():
        print("\n❌ راه‌اندازی ناموفق!")
        return False
    
    # گام 2: ایجاد جداول
    print("\n2️⃣  ایجاد جداول...")
    if not create_tables():
        print("\n❌ راه‌اندازی ناموفق!")
        return False
    
    print("\n" + "=" * 50)
    print("✅ دیتابیس با موفقیت راه‌اندازی شد!")
    print("=" * 50)
    print("\n💡 مراحل بعدی:")
    print("   1. فایل .env را با اطلاعات خود تنظیم کنید")
    print("   2. دستور 'pip install -r requirements.txt' را اجرا کنید")
    print("   3. سرور FastAPI را با 'uvicorn main:app --reload' شروع کنید")
    
    return True

if __name__ == "__main__":
    setup_database()
