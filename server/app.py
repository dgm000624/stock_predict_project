import yfinance as yf
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import kss
import mysql.connector
import logging
import traceback
import pytz
import os
import hashlib
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from datetime import datetime, timedelta
# BERT 분석을 위한 transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
# LLM 분석을 위한 Gemini API
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json # JSON 파싱을 위해 추가
import time
from db import get_conn
from threading import Thread
from train_models import train_and_predict_all_models # train_models.py는 같은 경로에 있어야 합니다.

# ==============================================================================
# 1. 초기 설정 및 모델 로드
# ==============================================================================

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', ping_timeout=60)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# DB_CONFIG: 사용자가 제공한 성공했던 DB 설정 사용
DB_CONFIG = {
    'host': '', 'user': '', 'password': '', 'database': 'stock_ai_db'
}

# 뉴스 분석 관련 전역 설정
GLOBAL_NEWS_CACHE = {} 
NEWS_API_KEY = "" #Newsapi.org API 
HUGGINGFACE_TOKEN = "" 
GEMINI_API_KEY = ""

GLOBAL_MODELS = {}
GLOBAL_GEMINI_MODEL = None # LLM 모델 전역 변수
SCORE_THRESHOLD = 0.6
MODEL_NAMES = [
    "snunlp/KR-FinBert-SC",
    "DataWizardd/finbert-sentiment-ko"
]


def initialize_gemini():
    """Gemini API 클라이언트를 초기화합니다."""
    global GLOBAL_GEMINI_MODEL, GEMINI_API_KEY
    
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_DEFAULT" or not GEMINI_API_KEY:
        print("🚨 경고: GEMINI_API_KEY가 설정되지 않았습니다. LLM 모드가 작동하지 않습니다.")
        return None
        
    try:
        # 테스트 코드와 동일하게 configure 호출
        genai.configure(api_key=GEMINI_API_KEY)
        # 모델만 생성
        GLOBAL_GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash') 
        print("✅ Gemini API 모델 초기화 완료.")
    except Exception as e:
        print(f"🚨 Gemini API 초기화 실패: {e}")
        GLOBAL_GEMINI_MODEL = None

def load_models():
    """BERT 모델을 로드합니다."""
    classifiers = {}
    try:
        tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAMES[0], token=HUGGINGFACE_TOKEN)
        model1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAMES[0], token=HUGGINGFACE_TOKEN)
        classifiers["classifier1"] = TextClassificationPipeline(model=model1, tokenizer=tokenizer1)
    except Exception as e:
        print(f"🚨 Model 1 로드 실패: {e}")
    try:
        tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAMES[1], token=HUGGINGFACE_TOKEN)
        model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAMES[1], token=HUGGINGFACE_TOKEN)
        classifiers["classifier2"] = TextClassificationPipeline(model=model2, tokenizer=tokenizer2)
    except Exception as e:
        print(f"🚨 Model 2 로드 실패: {e}")
    global GLOBAL_MODELS
    GLOBAL_MODELS = classifiers
    return classifiers

# ==============================================================================
# 2. DB 및 유틸리티 함수 (AI 예측 복구 및 뉴스 공통)
# ==============================================================================

def normalize_key(title: str) -> str:
    """제목을 안정적으로 캐싱하기 위한 해시 키 생성"""
    if not title: return ""
    normalized = re.sub(r'\s+', ' ', title.strip().lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def get_db_connection():
    try: return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"DB 연결 오류: {e}"); return None

def get_historical_data_from_db(ticker):
    print(f"--- DB에서 AI 예측 기록 조회: {ticker} ---")
    conn = get_db_connection()
    if not conn: return {}
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT model_name, target_date, actual_price, predicted_price FROM model_prediction_detail WHERE stock_code = %s ORDER BY target_date"
        cursor.execute(query, (ticker,))
        results = cursor.fetchall()
        if not results: return {}

        data_by_date = OrderedDict()
        model_names = sorted(list(set(r['model_name'] for r in results)))
        
        for row in results:
            if any(v is None for v in [row['target_date'], row['actual_price']]): continue
            date_str = row['target_date'].strftime('%Y-%m-%d')
            if date_str not in data_by_date:
                data_by_date[date_str] = {'actual': float(row['actual_price']), 'preds': {m: None for m in model_names}}
            if row['predicted_price'] is not None:
                data_by_date[date_str]['preds'][row['model_name']] = float(row['predicted_price'])

        dates = list(data_by_date.keys())
        actuals = [d['actual'] for d in data_by_date.values()]
        predictions = {model: [data_by_date[d]['preds'].get(model) for d in dates] for model in model_names}

        cursor.execute("SELECT test_start_index FROM model_comparison_log WHERE stock_code = %s LIMIT 1", (ticker,))
        log_result = cursor.fetchone()
        
        return {
            'dates': dates, 'actuals': actuals, 'predictions': predictions,
            'test_start_index': log_result.get('test_start_index', 0) if log_result else 0
        }
    except Exception as e:
        print(f"DB 조회 중 예외: {e}"); return {}
    finally:
        if conn and conn.is_connected(): conn.close()

# get_stock_data_from_yfinance: (기존 코드 유지)
def get_stock_data_from_yfinance(ticker, period="3y"):
    print(f"--- yfinance 단순 데이터 조회: {ticker} (기간: {period}) ---")
    try:
        interval = '1m' if period == '1d' else '1d'
        data_yf = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, group_by='ticker')
        if data_yf.empty: return None

        if isinstance(data_yf.columns, pd.MultiIndex):
            data_yf.columns = data_yf.columns.droplevel(0)

        if 'Close' not in data_yf.columns:
            if len(data_yf.columns) >= 4:
                data_yf.rename(columns={data_yf.columns[3]: 'Close'}, inplace=True)
            else: return None

        if period == '1d':
            dates = data_yf.index.strftime('%H:%M:%S').tolist()
        else:
            dates = data_yf.index.strftime('%Y-%m-%d').tolist()

        prices = data_yf['Close'].tolist()
        
        latest_price = prices[-1] if prices else 0
        latest_time = dates[-1] if dates else "N/A"

        return {'dates': dates, 'prices': prices, 'latest_price': latest_price, 'latest_time': latest_time}
    except Exception:
        traceback.print_exc(); return None

# get_and_cache_stock_names: (기존 코드 유지)
def get_and_cache_stock_names(tickers):
    names = {}
    conn = get_db_connection()
    if not conn:
        for ticker in tickers:
            try: names[ticker] = yf.Ticker(ticker).info.get('longName', ticker)
            except: names[ticker] = ticker
        return names

    try:
        cursor = conn.cursor(dictionary=True)
        format_strings = ','.join(['%s'] * len(tickers))
        cursor.execute(f"SELECT stock_code, stock_name FROM stock_info WHERE stock_code IN ({format_strings})", tuple(tickers))
        for row in cursor.fetchall(): names[row['stock_code']] = row['stock_name']
        
        missing_tickers = [t for t in tickers if t not in names]
        if missing_tickers:
            print(f"DB에 없는 종목 정보 조회: {missing_tickers}")
            for ticker in missing_tickers:
                try:
                    info = yf.Ticker(ticker).info
                    name = info.get('longName', ticker)
                    names[ticker] = name
                    insert_query = "INSERT INTO stock_info (stock_code, stock_name, industry, market_type) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE stock_name=VALUES(stock_name)"
                    cursor.execute(insert_query, (ticker, name, info.get('sector', 'N/A'), info.get('exchange', 'N/A')))
                except Exception as e:
                    print(f"yfinance에서 {ticker} 정보 조회 실패: {e}")
                    names[ticker] = ticker
            conn.commit()
    except Exception as e:
        print(f"종목 이름 조회 중 오류: {e}")
    finally:
        if conn.is_connected(): conn.close()
    return names

def initialize_stock_data():
    print(f"[{datetime.now()}] 🚀 초기 종목 정보 캐싱 시작...")
    target_tickers = get_all_tickers_from_db()
    default_tickers = ['005930.KS', 'AAPL', 'TSLA']
    target_tickers.extend(t for t in default_tickers if t not in target_tickers)
    if not target_tickers:
        print("경고: 초기화할 종목이 없습니다.")
        return
    names = get_and_cache_stock_names(target_tickers)
    print(f"✅ 초기 종목 정보 캐싱 완료. 로드된 종목 수: {len(names)}")

def get_all_tickers_from_db():
    tickers = []
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT stock_code FROM stock_info")
        tickers = [row[0] for row in cursor.fetchall()]
    except Exception:
        pass
    finally:
        if conn.is_connected(): conn.close()
    return tickers

# ==============================================================================
# 3. 뉴스 분석 및 스크래핑 (BERT & LLM 공통)
# ==============================================================================

def fetch_korean_news(query, page_size=20):
    url = "https://newsapi.org/v2/everything"
    # 기간 설정: 현재는 최근 7일(1주)
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    params = {"q": query, "language": "ko", "pageSize": page_size, "sortBy": "publishedAt", "apiKey": NEWS_API_KEY, "from": from_date}
    articles = []
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "content": article.get("content", "") or article.get("description", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", "")
            })
        return articles
    except Exception as e:
        print(f"뉴스 API 호출 오류: {e}")
        return []

def fetch_full_article(url):
    """기사 URL에서 본문 텍스트를 가져옵니다."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        return text
    except Exception:
        return ""

def is_valid_sentence(s):
    return re.search(r"[가-힣]", s)

# ==============================================================================
# 4. BERT 분석 함수
# ==============================================================================

def map_label(result):
    label = result["label"].upper()
    score = result.get("score",0)
    if "LABEL_2" in label or "POSITIVE" in label: mapped_label = "POSITIVE"
    elif "LABEL_0" in label or "NEGATIVE" in label: mapped_label = "NEGATIVE"
    else: mapped_label = "NEUTRAL"
    return mapped_label if score >= SCORE_THRESHOLD else "NEUTRAL"

def analyze_sentiment(sentence, models):
    if not models:
        return {"final_label":"NEUTRAL","model1_label":"N/A","model2_label":"N/A","model1_score":0.0,"model2_score":0.0}
    data = {"model1_label":"NEUTRAL","model1_score":0.0,"model2_label":"NEUTRAL","model2_score":0.0,"final_label":"NEUTRAL"}
    if "classifier1" in models:
        r1 = models["classifier1"](sentence)[0]
        data["model1_label"]=map_label(r1); data["model1_score"]=r1.get("score",0.0)
    if "classifier2" in models:
        r2 = models["classifier2"](sentence)[0]
        data["model2_label"]=map_label(r2); data["model2_score"]=r2.get("score",0.0)
    if data["model1_label"]!="NEUTRAL" and data["model1_label"]==data["model2_label"]: data["final_label"]=data["model1_label"]
    elif "classifier1" in models and "classifier2" not in models: data["final_label"]=data["model1_label"]
    elif "classifier2" in models and "classifier1" not in models: data["final_label"]=data["model2_label"]
    return data


# ==============================================================================
# 5. LLM (Gemini) 분석 함수
# ==============================================================================

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_label": {"type": "string", "description": "기사의 종합적인 감성 ('POSITIVE', 'NEGATIVE', 'NEUTRAL' 중 하나)"},
        "summary": {"type": "string", "description": "기사의 핵심 내용을 1~2문장으로 요약"},
        "key_sentiment_points": {"type": "array", "description": "감성을 판단한 주요 근거 2~3가지", "items": {"type": "string"}}
    },
    "required": ["final_label", "summary", "key_sentiment_points"]
}

# app.py (analyze_sentiment_with_llm 함수 내부)

def analyze_sentiment_with_llm(ticker, name, full_text):
    global GLOBAL_GEMINI_MODEL
    if GLOBAL_GEMINI_MODEL is None:
        return {'status': 'error', 'message': 'Gemini 모델이 초기화되지 않았습니다. API 키를 확인하세요.'}

    if len(full_text) < 100:
        return {'status': 'error', 'message': '기사 본문이 너무 짧아 (100자 미만) LLM 분석을 수행할 수 없습니다.'}

    # ★ 1. 프롬프트 내에서 JSON 응답 형식을 강제합니다.
    prompt = f"""
    당신은 금융 전문가입니다. 종목 {name} ({ticker})에 대한 다음 뉴스 기사를 분석하세요.
    
    1. 이 기사가 {ticker}의 주가에 미칠 영향의 **종합적인 감성**을 'POSITIVE', 'NEGATIVE', 'NEUTRAL' 중 하나로 판단하세요.
    2. 기사의 **핵심 내용**을 1~2문장으로 요약하세요.
    3. 해당 감성을 판단한 **주요 근거** 2~3가지를 찾으세요.

    분석 결과는 **반드시 다음 JSON 형식 문자열**로만 응답해야 합니다:
    {{
        "final_label": "POSITIVE 또는 NEGATIVE 또는 NEUTRAL",
        "summary": "1~2문장 요약",
        "key_sentiment_points": ["근거 1", "근거 2", "근거 3"]
    }}

    --- 기사 전문 ---
    {full_text}
    """
    
    try:
        # ★ 2. API 호출 시 인자를 최소화하고 safety_settings를 직접 전달합니다.
        # 구 버전은 config=나 response_mime_type을 지원하지 않습니다.
        response = GLOBAL_GEMINI_MODEL.generate_content(
            prompt,
            safety_settings=[ # 딕셔너리 리스트 형태로 전달
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            ]
        )
        
        # ★ 1. 마크다운 코드 블록 구문 제거
        raw_text = response.text.strip()
        
        # ```json\n으로 시작하고 ```로 끝나는 경우, 해당 구문을 제거합니다.
        if raw_text.startswith("```json"):
            # 첫 줄의 ```json\n 제거
            raw_text = raw_text.lstrip("```json\n")
        if raw_text.endswith("```"):
            # 마지막 줄의 ``` 제거
            raw_text = raw_text.rstrip("```")
            
        # 2. 순수한 JSON 문자열을 파싱
        llm_result = json.loads(raw_text.strip())
        llm_result['status'] = 'success'
        return llm_result

    except Exception as e:
        # JSON 파싱 실패 시 모델이 형식을 지키지 않은 것일 수 있음
        print(f"Gemini API 호출 중 오류: {e}")
        print(f"Gemini Raw Response Text: {response.text[:200]}...")
        return {'status': 'error', 'message': f'Gemini API 분석 오류: {e}. (응답 텍스트 확인 필요)'}

# ==============================================================================
# 6. Flask 라우트 (AI 예측 복구 및 뉴스 분석 분기)
# ==============================================================================
@app.route('/get_all_industries', methods=['GET'])
def get_all_industries():
    conn = get_db_connection()
    if not conn:
        return jsonify({'status': 'error', 'message': 'DB connection failed'}), 500
    try:
        cursor = conn.cursor(dictionary=True)
        # 'N/A'가 아니거나 비어있지 않은 유효한 산업 목록만 조회
        cursor.execute("SELECT DISTINCT industry FROM stock_info WHERE industry IS NOT NULL AND industry != 'N/A' AND industry != '' ORDER BY industry")
        industries = [row['industry'] for row in cursor.fetchall()]
        return jsonify({'status': 'success', 'industries': industries})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn and conn.is_connected(): conn.close()

# [추가] 사용자가 선택한 산업의 주식 목록을 가져오는 API
@app.route('/get_stocks_by_industry', methods=['POST'])
def get_stocks_by_industry():
    data = request.json
    industry = data.get('industry')
    if not industry:
        return jsonify({'status': 'error', 'message': 'Industry required'}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'status': 'error', 'message': 'DB connection failed'}), 500
    try:
        cursor = conn.cursor(dictionary=True)
        # 해당 산업의 주식을 10개까지 조회
        query = "SELECT stock_code, stock_name FROM stock_info WHERE industry = %s LIMIT 10"
        cursor.execute(query, (industry,))
        stocks = cursor.fetchall()
        return jsonify({'status': 'success', 'stocks': stocks})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn and conn.is_connected(): conn.close()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_simple_chart_data', methods=['POST'])
def get_simple_chart_data():
    data = request.json
    ticker = data.get('ticker')
    period = data.get('period', '1y')
    
    chart_data = get_stock_data_from_yfinance(ticker, period=period)
    
    if chart_data:
        return jsonify(chart_data)
    else:
        return jsonify({'status': 'error', 'message': f"'{ticker}' 데이터를 가져오는 데 실패했습니다."}), 404


@app.route('/get_current_price', methods=['POST'])
def get_current_price():
    ticker = request.json.get('ticker')
    if not ticker:
        return jsonify({'price': None, 'time': 'N/A'})

    try:
        data = yf.download(ticker, period='1d', interval='1m', progress=False, auto_adjust=True)
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            latest_time_utc = data.index[-1].tz_convert('UTC')
            kst = pytz.timezone('Asia/Seoul')
            latest_time_kst = latest_time_utc.astimezone(kst)

            # 성공했던 코드로 복구
            return jsonify({'price': float(latest_price), 'time': latest_time_kst.strftime('%Y-%m-%d %H:%M:%S')})
    except Exception as e:
        print(f"현재가 조회 오류: {e}")

    return jsonify({'price': None, 'time': 'Error'})

@app.route('/get_stock_names', methods=['POST'])
def get_stock_names_api():
    data = request.json
    tickers = data.get('tickers', [])
    if not tickers:
        return jsonify({})
    names = get_and_cache_stock_names(tickers)
    return jsonify(names)

@app.route('/switch_ticker', methods=['POST'])
def switch_ticker():
    data = request.get_json(silent=True) or {}
    ticker = (data.get('ticker') or '').strip()
    historical_days = int(data.get('historical_days', 365))

    if not ticker:
        return jsonify({'status': 'error', 'message': 'ticker is required'}), 400

    def run_training(tk, hist_days):
        try:
            app.logger.info(f"[BG] training start: {tk}")
            train_and_predict_all_models(ticker=tk, historical_days=hist_days)
            socketio.emit('training_complete', {'status': 'success', 'ticker': tk})
        except Exception as e:
            app.logger.exception(f"[BG] training error: {e}")
            socketio.emit('training_error', {'status': 'error', 'ticker': tk, 'message': str(e)})

    socketio.start_background_task(run_training, ticker, historical_days)
    return jsonify({'status': 'ok', 'message': f"'{ticker}' 분석을 시작합니다."}), 202


@app.route('/get_ai_results', methods=['POST'])
def get_ai_results():
    data = request.json
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'status': 'error', 'message': 'Ticker is required'}), 400
    
    print(f"--- 웹 브라우저의 요청에 따라 AI 예측 기록 조회: {ticker} ---")
    historical_data = get_historical_data_from_db(ticker)
    
    if historical_data and historical_data.get('dates'):
        return jsonify(historical_data)
    else:
        # 성공했던 코드로 복구
        return jsonify({'status': 'error', 'message': 'DB에서 분석 결과를 가져오지 못했습니다.'}), 404

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    'AI 예측 상승률 Top 5'를 반환합니다.
    (모든 'base' 모델의 평균 예측가 - 최신 실제 종가) / 최신 실제 종가
    """
    
    top_movers = []
    conn = get_db_connection()
    if not conn:
        return jsonify({'status': 'error', 'message': 'DB connection failed'}), 500

    try:
        cursor = conn.cursor(dictionary=True)
        
        # AI 예측 상승률 Top 5를 계산하는 SQL 쿼리 (모델 평균 사용)
        query = """
            WITH NextPredictions AS (
                -- 1. 각 종목별 'base' 모델의 '평균' 예측값을 계산합니다.
                SELECT
                    stock_code,
                    AVG(predicted_price) as avg_predicted_price
                FROM future_predictions
                WHERE prediction_date >= CURRENT_DATE()
                  AND variant = 'base'
                GROUP BY stock_code
            ),
            LastActualPrice AS (
                -- 2. 각 종목의 가장 최신 실제 종가를 가져옵니다.
                SELECT
                    stock_code,
                    close AS last_close,
                    ROW_NUMBER() OVER(PARTITION BY stock_code ORDER BY date DESC) as rn
                FROM daily_price
            ),
            PredictedMovers AS (
                -- 3. 평균 예측값과 실제 종가를 비교하여 '평균 예측 상승률'을 계산합니다.
                SELECT
                    p.stock_code,
                    a.last_close,
                    p.avg_predicted_price,
                    ((p.avg_predicted_price - a.last_close) / a.last_close) * 100 AS predicted_change_percent
                FROM NextPredictions p
                JOIN LastActualPrice a ON p.stock_code = a.stock_code
                WHERE a.rn = 1 AND a.last_close > 0 AND p.avg_predicted_price IS NOT NULL
            )
            -- 4. 상승률이 0%보다 큰 종목만 Top 5를 조회합니다.
            SELECT
                m.stock_code,
                s.stock_name,
                m.predicted_change_percent
            FROM PredictedMovers m
            LEFT JOIN stock_info s ON m.stock_code = s.stock_code
            WHERE m.predicted_change_percent > 0
            ORDER BY m.predicted_change_percent DESC
            LIMIT 5
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        for row in results:
            top_movers.append({
                'ticker': row['stock_code'],
                'name': row['stock_name'] or row['stock_code'],
                'change': f"{row['predicted_change_percent']:.2f}%"
            })
            
    except Exception as e:
        print(f"AI 랭킹(모델 평균) 기반 추천 DB 조회 중 오류: {e}")
    finally:
        if conn and conn.is_connected(): conn.close()
    
    return jsonify({
        'top_movers': top_movers
    })

@app.route('/analyze_news_sentiment', methods=['POST'])
def analyze_news_sentiment_api():
    start_total = time.time()

    data = request.json
    ticker = data.get('ticker')
    requested_index = data.get('index', 0) 
    mode = data.get('mode', 'bert') # ★ mode 파라미터 추가

    global GLOBAL_NEWS_CACHE
    if ticker not in GLOBAL_NEWS_CACHE:
        GLOBAL_NEWS_CACHE[ticker] = {}

    if mode == 'bert' and not GLOBAL_MODELS:
        return jsonify({'status': 'error', 'message': 'BERT 모델 미로드'}), 503
    if mode == 'llm' and GLOBAL_GEMINI_MODEL is None:
        print("🚨 LLM 분석 요청 실패: Gemini 모델이 초기화되지 않았거나 API 키에 문제가 있습니다.")
        return jsonify({'status': 'error', 'message': 'Gemini API 키가 설정되지 않았거나 모델 로드에 실패했습니다. (503)'}), 503


    # 1️⃣ 종목 이름 조회
    stock_names = get_and_cache_stock_names([ticker])
    name = stock_names.get(ticker, ticker)

    # 2️⃣ 뉴스 가져오기 및 필터링
    if ticker in ['AAPL', 'TSLA', 'MSFT', 'GOOGL']:
        query = ticker
    else:
        short_name = re.sub(r'\([^)]*\)|\s*유가증권|코스피|코스닥|\s*\(KOSPI\)|\s*\(KOSDAQ\)', '', name).strip()
        query = short_name if short_name else name
    
    articles = fetch_korean_news(query=query, page_size=20) 
    
    if not articles:
        return jsonify({
            'status': 'error', 
            'message': 'News API로부터 종목 관련 기사를 전혀 가져오지 못했습니다. (API 키/할당량 초과 가능성)',
            'total_valid_news': 0
        }), 404


    articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
    seen = set()
    unique_articles = []
    for a in articles:
        if a['title'] not in seen:
            unique_articles.append(a)
            seen.add(a['title'])
    articles = unique_articles
    
    valid_count = 0
    target_raw_index = -1
    target_article = None
    
    for raw_index, article in enumerate(articles):
        title = article.get('title', '')
        key = normalize_key(title)
        
        is_valid = False
        sentences_to_analyze = []

        # 캐시 키에 mode를 포함하여 BERT와 LLM 결과를 분리 저장
        cache_key_with_mode = f"{key}_{mode}"
        
        if cache_key_with_mode in GLOBAL_NEWS_CACHE[ticker]:
            is_valid = True
            
        else:
            text = article.get('content') or article.get('description') or ""
            
            if re.search(r'\[\+\d+ chars\]', text):
                 text = re.sub(r'\[\+\d+ chars\]', '', text)
            if not is_valid_sentence(text) and article.get('url'):
                text = fetch_full_article(article['url'])
            
            cleaned_text = re.sub(r'\[\+\d+ chars\]', '', text)
            
            # BERT는 문장 단위 분리, LLM은 전체 텍스트 사용
            if mode == 'bert':
                sentences = kss.split_sentences(cleaned_text[:2000]) 
                sentences_to_analyze = [s for s in sentences if s.strip() and is_valid_sentence(s)]
                is_valid = len(sentences_to_analyze) > 0
            elif mode == 'llm':
                 # LLM은 100자 미만 기사만 제외하고 유효하다고 간주
                 is_valid = len(cleaned_text.strip()) >= 100
            
            if not is_valid: continue 

        valid_count += 1
        
        if valid_count == requested_index + 1:
            target_raw_index = raw_index
            target_article = article
            # 캐시 미스였을 경우, 분석에 사용할 텍스트를 기사에 임시 저장
            if not is_valid:
                target_article['temp_cleaned_text'] = cleaned_text
                target_article['temp_sentences'] = sentences_to_analyze
            break
            
    if target_raw_index == -1:
        # 유효 기사를 찾지 못했을 때 특정 메시지 반환
        return jsonify({
            'status': 'error', 
            'message': '현재 선택된 종목에 대해 분석할 유효한 뉴스를 찾지 못했습니다. 기간이나 검색어를 조정해 보세요.',
            'total_valid_news': valid_count
        }), 404

    article = target_article
    title = article.get('title', '')
    key = normalize_key(title)
    cache_key_with_mode = f"{key}_{mode}"
    
    if cache_key_with_mode in GLOBAL_NEWS_CACHE[ticker]:
        cached_result = GLOBAL_NEWS_CACHE[ticker][cache_key_with_mode].copy()
        cached_result['status'] = 'success (cached)'
        cached_result['analyzed_index'] = target_raw_index
        return jsonify(cached_result)

    # B. 분석 수행
    try:
        # 1. 텍스트 추출
        if 'temp_cleaned_text' in article:
            cleaned_text = article['temp_cleaned_text']
            sentences_to_analyze = article.get('temp_sentences', [])
        else:
            text = article.get('content') or article.get('description') or ""
            if re.search(r'\[\+\d+ chars\]', text): text = re.sub(r'\[\+\d+ chars\]', '', text)
            if not is_valid_sentence(text) and article.get('url'): text = fetch_full_article(article['url'])
            cleaned_text = re.sub(r'\[\+\d+ chars\]', '', text)
            
            if mode == 'bert':
                cleaned_text = cleaned_text[:2000]
                sentences = kss.split_sentences(cleaned_text)
                sentences_to_analyze = [s for s in sentences if s.strip() and is_valid_sentence(s)]


        # 2. BERT 모드와 LLM 모드 분기 처리
        if mode == 'bert':
            all_results = [analyze_sentiment(s, GLOBAL_MODELS) for s in sentences_to_analyze]
            final_labels = [r['final_label'] for r in all_results if r['final_label'] != 'NEUTRAL']
            pos = final_labels.count('POSITIVE')
            neg = final_labels.count('NEGATIVE')
            overall = 'POSITIVE' if pos > neg else 'NEGATIVE' if neg > pos else 'NEUTRAL'
            avg1 = np.mean([r['model1_score'] for r in all_results]) if all_results else 0
            avg2 = np.mean([r['model2_score'] for r in all_results]) if all_results else 0

            analysis_result = {
                "final_label": overall,
                "model1_score": float(avg1),
                "model2_score": float(avg2),
                "positive_count": pos,
                "negative_count": neg,
                "analyzed_sentences": sentences_to_analyze[:5],
                "total_sentences": len(sentences_to_analyze),
                "analysis_mode": "BERT"
            }
        
        elif mode == 'llm':
            llm_response = analyze_sentiment_with_llm(ticker, name, cleaned_text)

            if llm_response.get('status') == 'error':
                return jsonify(llm_response), 500
            
            analysis_result = {
                "final_label": llm_response['final_label'],
                "summary": llm_response['summary'],
                "key_sentiment_points": llm_response['key_sentiment_points'],
                "analysis_mode": "LLM"
            }

        else:
             return jsonify({'status': 'error', 'message': '잘못된 분석 모드입니다.'}), 400
        
        # 3. 최종 결과 통합 및 캐시 저장
        result = {
            "status": "success",
            "ticker": ticker,
            "title": title,
            "source": article.get('source'),
            "url": article.get('url'),
            "full_original_text": cleaned_text,
            "analyzed_index": target_raw_index,
            **analysis_result 
        }

        GLOBAL_NEWS_CACHE[ticker][cache_key_with_mode] = result.copy()

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
#(ai전부 상승 예측)알림
from flask import request, jsonify

@app.post("/api/alerts/run")
def run_consensus_alerts():
    body = request.get_json(silent=True) or {}
    target_date  = body.get("date")
    direction    = (body.get("direction") or "up").lower()   # up|down|both
    min_conf     = body.get("min_confidence")                # None or float
    require_all  = bool(body.get("require_all", True))
    min_models   = body.get("min_models")                    # int or None
    alert_prefix = (body.get("alert_prefix") or "consensus").lower()  # "consensus"|"model"|...

    with get_conn() as conn, conn.cursor() as cur:
        # 최신 날짜 기본값
        if not target_date:
            cur.execute("SELECT MAX(date) AS d FROM prediction_result")
            row = cur.fetchone()
            target_date = row["d"]
            if not target_date:
                return jsonify({"ok": False, "msg": "prediction_result가 비어있음"}), 400

        # min_conf 필터 조각
        conf_sql = "AND (confidence IS NULL OR confidence >= %s)" if (min_conf is not None) else ""
        conf_param = [float(min_conf)] if (min_conf is not None) else []

        # (stock_code, date)별 up/down 카운트
        group_sql = f"""
            SELECT
                stock_code,
                date,
                COUNT(*) AS n_models,
                SUM(CASE WHEN UPPER(predict_class) IN ('UP','RISE','BUY','BULL') {conf_sql} THEN 1 ELSE 0 END) AS n_up,
                SUM(CASE WHEN UPPER(predict_class) IN ('DOWN','SELL','BEAR') {conf_sql} THEN 1 ELSE 0 END)      AS n_down
            FROM prediction_result
            WHERE date = %s
            GROUP BY stock_code, date
        """

        # 합의 조건식
        up_cond_sql = "g.n_up = g.n_models" if require_all else "g.n_up >= COALESCE(%s, CEIL(g.n_models/2))"
        dn_cond_sql = "g.n_down = g.n_models" if require_all else "g.n_down >= COALESCE(%s, CEIL(g.n_models/2))"

        inserted_total = 0

        def insert_by_direction(is_up: bool):
            nonlocal inserted_total
            alert_type   = f"{alert_prefix}_{'up' if is_up else 'down'}"
            dir_label    = "상승" if is_up else "하락"
            consensus_tag = " 합의" if alert_prefix == "consensus" else ""
            conf_tag     = f" (conf≥{min_conf})" if min_conf is not None else ""
            models_tag   = ""
            if not require_all:
                if min_models is not None:
                    models_tag = f" (≥{int(min_models)} models)"
                else:
                    models_tag = " (≥과반)"  # 과반 기본
            # 메시지 앞부분은 파이썬에서 만들고 SQL에서는 CONCAT(%s, GROUP_CONCAT(...))
            message_prefix = f"모델{consensus_tag} {dir_label}{conf_tag}{models_tag} | 참여모델: "

            cond_sql = up_cond_sql if is_up else dn_cond_sql

            sql = f"""
                INSERT INTO user_alerts (stock_code, date, alert_type, message)
                SELECT
                  pr.stock_code,
                  pr.date,
                  %s AS alert_type,
                  CONCAT(%s, GROUP_CONCAT(DISTINCT pr.model_name ORDER BY pr.model_name SEPARATOR ', ')) AS message
                FROM prediction_result pr
                JOIN (
                    {group_sql}
                ) g
                  ON g.stock_code = pr.stock_code AND g.date = pr.date
                LEFT JOIN user_alerts ua
                  ON ua.stock_code = pr.stock_code
                 AND ua.date = pr.date
                 AND ua.alert_type = %s
                WHERE pr.date = %s
                  AND {cond_sql}
                  AND ua.id IS NULL
                GROUP BY pr.stock_code, pr.date
            """

            params = [
                alert_type,            # alert_type
                message_prefix,        # message prefix
            ]
            # group_sql 파라미터들
            params += conf_param + conf_param + [target_date]
            # LEFT JOIN alert_type, pr.date
            params += [alert_type, target_date]
            # require_all=False면 min_models 바인딩
            if not require_all:
                params += [min_models]
            cur.execute(sql, params)
            inserted_total += cur.rowcount

        # 실행
        if direction in ("up", "both"):
            insert_by_direction(is_up=True)
        if direction in ("down", "both"):
            insert_by_direction(is_up=False)

        return jsonify({
            "ok": True,
            "date": str(target_date),
            "direction": direction,
            "require_all": require_all,
            "min_models": min_models,
            "min_confidence": min_conf,
            "inserted": inserted_total
        })

#알림 조회 라우트
@app.get("/api/alerts")
def list_alerts():
    stock_code = request.args.get("stock_code")
    date_from  = request.args.get("date_from")
    date_to    = request.args.get("date_to")
    a_type     = request.args.get("type")  # 예: consensus_up, consensus_down, model_up, model_down

    clauses, params = [], []
    if a_type:
        clauses.append("alert_type = %s"); params.append(a_type)
    if stock_code:
        clauses.append("stock_code = %s"); params.append(stock_code)
    if date_from:
        clauses.append("date >= %s"); params.append(date_from)
    if date_to:
        clauses.append("date <= %s"); params.append(date_to)

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    q = f"""
      SELECT id, stock_code, date, alert_type, message, created_at
      FROM user_alerts
      {where_sql}
      ORDER BY date DESC, stock_code
      LIMIT 500
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    return jsonify({"ok": True, "count": len(rows), "items": rows})

# ==============================================================================
# 7. 서버 초기화
# ==============================================================================
def initialize():
    load_models() # BERT 로드
    initialize_gemini() # Gemini 로드
    initialize_stock_data()
    print("✅ 모든 초기화 완료.")

if __name__ == '__main__':
    initialize()
    print("--- Eventlet 기반 고성능 서버를 시작합니다. http://127.0.0.1:5000 ---")
    socketio.run(app, host='0.0.0.0', port=5000)