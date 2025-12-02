# ingest_prices_from_file.py
# tickers.txt에서 티커를 읽어 yfinance로 일봉을 수집하여 MySQL stock_ai_db.daily_price에 UPSERT합니다.
# 사용법:
#   python ingest_prices_from_file.py --file tickers.txt --period 7d --eval
#   python ingest_prices_from_file.py --file tickers.txt --start 2025-01-01 --end 2025-10-26

import argparse
import sys
import time
from datetime import datetime, date
from typing import List, Iterable

import yfinance as yf
import mysql.connector
from pathlib import Path


# === DB 접속 정보 (환경에 맞게 수정) ===
DB = dict(
    host='localhost',
    user='',
    password='',
    database='stock_ai_db',
)

UPSERT_SQL = """
INSERT INTO daily_price
  (stock_code, date, open, high, low, close, volume, label)
VALUES
  (%s, %s, %s, %s, %s, %s, %s, NULL)
ON DUPLICATE KEY UPDATE
  open=VALUES(open),
  high=VALUES(high),
  low=VALUES(low),
  close=VALUES(close),
  volume=VALUES(volume)
"""

EVAL_SQL = """
UPDATE future_predictions f
LEFT JOIN daily_price dp
  ON dp.stock_code = f.stock_code AND dp.date = f.prediction_date
SET f.actual_price = COALESCE(f.actual_price, dp.close),
    f.abs_error    = CASE WHEN COALESCE(f.actual_price, dp.close) IS NULL
                          THEN NULL
                          ELSE ABS(COALESCE(f.actual_price, dp.close) - f.predicted_price) END,
    f.mape         = CASE WHEN COALESCE(f.actual_price, dp.close) IS NULL OR COALESCE(f.actual_price, dp.close)=0
                          THEN NULL
                          ELSE ABS((COALESCE(f.actual_price, dp.close)-f.predicted_price)/COALESCE(f.actual_price, dp.close))*100 END,
    f.evaluated_at = CASE WHEN dp.close IS NOT NULL THEN NOW() ELSE f.evaluated_at END
WHERE f.actual_price IS NULL
  AND f.prediction_date < CURRENT_DATE();
"""

def load_tickers(path: str) -> List[str]:
    """tickers.txt에서 티커를 읽어 정제된 리스트를 반환."""
    tickers: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # 쉼표/공백/탭 모두 구분자로 처리
            parts: Iterable[str] = []
            for chunk in line.replace(',', ' ').split():
                if chunk and not chunk.startswith('#'):
                    parts = list(parts) + [chunk]
            for t in parts:
                t = t.strip()
                if t:
                    tickers.append(t)
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def fetch_and_upsert(conn, ticker: str, period: str = None, start: str = None, end: str = None, sleep_sec: float = 0.8) -> int:
    """단일 티커에 대해 yfinance 일봉을 가져와 daily_price에 UPSERT. 반환: upsert 건수"""
    cur = conn.cursor()
    # yfinance 호출
    try:
        if period:
            hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        else:
            hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    except Exception as e:
        print(f"[WARN] yfinance error for {ticker}: {e}")
        cur.close()
        return 0

    if hist is None or hist.empty:
        print(f"[WARN] no data for {ticker}")
        cur.close()
        return 0

    n = 0
    for ts, row in hist.iterrows():
        try:
            d = ts.to_pydatetime().date() if hasattr(ts, "to_pydatetime") else ts.date()
        except Exception:
            # 혹시 모를 타입 이슈 방어
            d = date.fromisoformat(str(ts)[:10])

        # 각 컬럼 안전 캐스팅
        def fget(name, default=0.0):
            val = row.get(name, default)
            try:
                return float(val)
            except Exception:
                try:
                    # 시리즈/ndarray 등일 때 첫 원소
                    return float(getattr(val, "iloc", [val])[0])
                except Exception:
                    return float(default)

        def iget(name, default=0):
            val = row.get(name, default)
            try:
                return int(val)
            except Exception:
                try:
                    return int(getattr(val, "iloc", [val])[0])
                except Exception:
                    return int(default)

        args = (
            ticker, d,
            fget("Open"), fget("High"), fget("Low"), fget("Close"),
            iget("Volume"),
        )
        try:
            cur.execute(UPSERT_SQL, args)
            n += 1
        except mysql.connector.Error as db_err:
            print(f"[ERROR] UPSERT failed for {ticker} {d}: {db_err}")
            # 계속 진행
        except Exception as e:
            print(f"[ERROR] unexpected UPSERT error for {ticker} {d}: {e}")

    conn.commit()
    cur.close()

    # 간단한 레이트 리밋 완화
    time.sleep(sleep_sec)
    return n

DEFAULT_TICKERS = Path(__file__).parent / "tickers.txt"

def parse_args():
    p = argparse.ArgumentParser(description="Ingest daily closes into stock_ai_db.daily_price from tickers.txt")
    p.add_argument("--file", "-f",
               default=str(DEFAULT_TICKERS),
               help="tickers.txt 경로")
    # 기간 지정(간단): period 또는 start/end 중 하나를 사용
    p.add_argument("--period", default="7d", help="yfinance period (예: 7d, 14d, 1mo, 3mo). start/end 대신 사용")
    p.add_argument("--start", help="YYYY-MM-DD (period 대신)")
    p.add_argument("--end", help="YYYY-MM-DD (period 대신)")
    p.add_argument("--eval", action="store_true", help="수집 후 예측 평가 UPDATE까지 수행")
    p.add_argument("--sleep", type=float, default=0.8, help="티커 간 대기(초), rate limit 완화용")
    return p.parse_args()

def main():
    args = parse_args()

    tickers = load_tickers(args.file)
    if not tickers:
        print("[ERROR] No tickers found in file.")
        sys.exit(1)

    # period vs start/end 충돌 방지
    use_period = args.period and not (args.start or args.end)
    if not use_period and not (args.start and args.end):
        print("[ERROR] 기간 지정이 올바르지 않습니다. --period 또는 (--start & --end) 중 하나를 사용하세요.")
        sys.exit(1)

    conn = mysql.connector.connect(**DB)
    cur = conn.cursor()
    cur.execute("USE stock_ai_db")

    total_rows = 0
    print(f"[INFO] ingest start: {len(tickers)} tickers")
    for tk in tickers:
        if use_period:
            n = fetch_and_upsert(conn, tk, period=args.period, sleep_sec=args.sleep)
        else:
            n = fetch_and_upsert(conn, tk, start=args.start, end=args.end, sleep_sec=args.sleep)
        total_rows += n
        print(f"[OK] {tk}: upsert {n} rows")

    if args.eval:
        print("[INFO] evaluating predictions (backfill next-day actuals)...")
        try:
            cur.execute(EVAL_SQL)
            conn.commit()
            print("[OK] evaluation backfill done.")
        except Exception as e:
            print(f"[WARN] evaluation failed: {e}")

    cur.close()
    conn.close()
    print(f"[DONE] total upsert rows: {total_rows}")

if __name__ == "__main__":
    main()
