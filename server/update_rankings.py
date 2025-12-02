import yfinance as yf
import mysql.connector
from datetime import datetime
import pandas as pd

# --- [1] 기본 설정 ---
DB_CONFIG = { 'host': 'localhost', 'user': '', 'password': '', 'database': 'stock_ai_db' }

# --- [2] 필요한 함수들 ---

def get_all_tickers_from_db():
    """DB의 stock_info 테이블에서 모든 종목 코드를 가져옵니다."""
    tickers = []
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT stock_code FROM stock_info")
        results = cursor.fetchall()
        tickers = [row[0] for row in results]
        print(f"DB에서 총 {len(tickers)}개의 종목을 랭킹 후보로 가져왔습니다.")
    except Exception as e:
        print(f"DB에서 종목 코드 조회 중 오류: {e}")
    finally:
        if conn.is_connected(): conn.close()
    return tickers

def get_and_cache_stock_names(tickers):
    """
    주어진 티커 목록의 이름을 조회합니다.
    1. DB에서 먼저 찾아보고, 2. 없으면 yfinance에 물어본 뒤, 3. 그 결과를 다시 DB에 저장합니다.
    """
    names = {}
    conn = mysql.connector.connect(**DB_CONFIG)
    if not conn:
        for ticker in tickers:
            try:
                names[ticker] = yf.Ticker(ticker).info.get('longName', ticker)
            except:
                names[ticker] = ticker
        return names

    try:
        cursor = conn.cursor(dictionary=True)
        if tickers:
            format_strings = ','.join(['%s'] * len(tickers))
            query = f"SELECT stock_code, stock_name FROM stock_info WHERE stock_code IN ({format_strings})"
            cursor.execute(query, tuple(tickers))
            for row in cursor.fetchall():
                if row['stock_name']:
                    names[row['stock_code']] = row['stock_name']
        
        missing_tickers = [t for t in tickers if t not in names]
        if missing_tickers:
            print(f"DB에 없는 종목 이름 조회: {missing_tickers}")
            for ticker in missing_tickers:
                try:
                    info = yf.Ticker(ticker).info
                    name = info.get('longName', ticker)
                    names[ticker] = name
                    insert_query = "INSERT INTO stock_info (stock_code, stock_name, industry) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE stock_name=VALUES(stock_name)"
                    cursor.execute(insert_query, (ticker, name, info.get('sector', 'N/A')))
                except Exception as e:
                    print(f"yfinance에서 {ticker} 정보 조회 실패: {e}")
                    names[ticker] = ticker
            conn.commit()
    except Exception as e:
        print(f"종목 이름 조회 중 오류: {e}")
    finally:
        if conn.is_connected(): conn.close()
    return names

def update_top_5_ranking():
    """DB의 모든 종목을 대상으로 상승률 Top 5를 계산하고, 그 결과를 DB에 저장합니다."""
    print(f"[{datetime.now()}] 🚀 랭킹 업데이트를 시작합니다...")
    
    target_tickers = get_all_tickers_from_db()
    if not target_tickers:
        print("랭킹을 계산할 대상 종목이 DB에 없습니다.")
        return

    try:
        full_data = yf.download(target_tickers, period="2d", progress=False)

        if full_data.empty:
            print("yfinance에서 데이터를 가져오지 못했습니다.")
            return
            
        close_prices = full_data['Close']
        if len(close_prices) < 2:
            print("데이터가 부족하여 랭킹을 업데이트할 수 없습니다.")
            return

        last_day = close_prices.iloc[-1]
        prev_day = close_prices.iloc[-2]
        change = ((last_day - prev_day) / prev_day * 100).dropna().sort_values(ascending=False)
        
        top_5_series = change.head(5)
        top_5_tickers = top_5_series.index.tolist()

        top_5_names = get_and_cache_stock_names(top_5_tickers)
        print(f"오늘의 Top 5: {top_5_names}")

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM daily_ranking")

        for i, ticker in enumerate(top_5_tickers):
            perc = top_5_series[ticker]
            stock_name = top_5_names.get(ticker, ticker)
            
            insert_query = "INSERT INTO daily_ranking (rank_order, stock_code, stock_name, change_percent, updated_at) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(insert_query, (i + 1, ticker, stock_name, perc, datetime.now()))
        
        conn.commit()
        print("✅ 새로운 Top 5 랭킹을 데이터베이스에 성공적으로 저장했습니다.")

    except Exception as e:
        print(f"랭킹 업데이트 중 오류 발생: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    update_top_5_ranking() 