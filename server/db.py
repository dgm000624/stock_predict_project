# db.py
import os, pymysql

def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", ""),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "stock_ai_db"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )
