import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import gc
import logging
from datetime import date, timedelta, datetime
import json
import mysql.connector as mysql
import os, json, requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_models")

DB_CONFIG = {
    'host': '',
    'user': '',
    'password': '',
    'database': 'stock_ai_db'
}


def get_db_connection():
    try:
        conn = mysql.connect(**DB_CONFIG)
        return conn
    except mysql.Error as e:
        # DB 연결 오류 시 로그 출력 및 None 반환
        logger.error(f"데이터베이스 연결 오류: {e.msg}")
        return None

def get_and_store_stock_data(ticker, days_back=1095):
    conn = get_db_connection()
    if not conn: return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT stock_code FROM stock_info WHERE stock_code = %s", (ticker,))
        if not cursor.fetchone():
            try:
                logger.info(f"'{ticker}' 정보가 stock_info 테이블에 없어 yfinance에서 조회 후 추가합니다.")
                ticker_info = yf.Ticker(ticker).info
                insert_stock_info_query = "INSERT INTO stock_info (stock_code, stock_name, industry, market_type) VALUES (%s, %s, %s, %s)"
                cursor.execute(insert_stock_info_query, (ticker, ticker_info.get('longName', ticker), ticker_info.get('sector', 'N/A'), ticker_info.get('exchange', 'N/A')))
                conn.commit()
            except Exception as e:
                logger.error(f"yfinance에서 '{ticker}' 정보를 가져오거나 저장하는 중 오류: {e}")
                return None
        
        end_date, start_date = date.today(), date.today() - timedelta(days=days_back)
        query = "SELECT date, open, high, low, close, volume FROM daily_price WHERE stock_code = %s AND date BETWEEN %s AND %s ORDER BY date"
        cursor.execute(query, (ticker, start_date, end_date))
        data = cursor.fetchall()
        
        if len(data) > (days_back / 365) * 252 * 0.8:
            logger.info(f"'{ticker}' 데이터를 DB에서 성공적으로 로드했습니다.")
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)
            return df

        logger.info(f"'{ticker}' 데이터가 DB에 부족하여 yfinance에서 다운로드 후 저장합니다.")
        data_yf = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data_yf.empty:
            logger.error(f"CRITICAL: yfinance가 '{ticker}'에 대한 데이터를 반환하지 않았습니다.")
            return None

        # 결측치(NaN) 행을 간단하게 제거하여 안정성을 높입니다.
        data_yf.dropna(inplace=True)

        logger.info(f"yfinance에서 '{ticker}' 데이터 {len(data_yf)}개를 성공적으로 다운로드했습니다.")

        insert_query = "INSERT INTO daily_price (stock_code, date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close), volume=VALUES(volume)"
        data_to_insert = [(ticker, idx.date(), float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), int(row['Volume'])) for idx, row in data_yf.iterrows()]
        
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        return data_yf
    except Exception as e:
        logger.error(f"get_and_store_stock_data 함수에서 예외 발생: {e}", exc_info=True)
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()

def train_and_predict_all_models(
    ticker,
    historical_days=1095,
    n_steps=60,
    test_size=0.2,
    params_by_model=None,
    run_tag=None,
    variant="base",          # ★ variant 인자 유지
    as_of_date=None          # (선택) 마지막 장일 고정 시 사용
):
    
    logger.info(f"'{ticker}' 분석 시작: 데이터 로딩 및 전처리...")
    stock_data = get_and_store_stock_data(ticker, days_back=historical_days)
    if stock_data is None or stock_data.empty:
        logger.error(f"'{ticker}'에 대한 주가 데이터를 가져오지 못했습니다.")
        return None

    if stock_data.isnull().sum().sum() > 0:
        stock_data.fillna(method='ffill', inplace=True)
        stock_data.fillna(method='bfill', inplace=True)

    close_prices = stock_data['Close']
    df_features = pd.DataFrame({'Close': close_prices.values.flatten()}, index=close_prices.index)
    close = df_features['Close']
    lag_list = [close.shift(i).rename(f'lag_{i}') for i in range(1, n_steps + 1)]
    lag_block = pd.concat(lag_list, axis=1)
    df_features = pd.concat([df_features, lag_block], axis=1)
    df_features = df_features.copy()
    df_features = df_features.dropna()
    if df_features.empty:
        logger.error("데이터 전처리 후 남은 데이터가 없습니다. (기간이 너무 짧을 수 있습니다)")
        return None

    y_true = df_features[['Close']]
    X_features = df_features.drop('Close', axis=1)
    split_index = int(len(X_features) * (1 - test_size))
    X_train, X_test = X_features[:split_index], X_features[split_index:]
    y_train, y_test = y_true[:split_index], y_true[split_index:]

    trained_models = {}
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        # 기존 기록 정리(원하면 유지/주석 처리 가능)
        cursor.execute("DELETE FROM model_comparison_log WHERE stock_code = %s", (ticker,))
        cursor.execute("DELETE FROM model_prediction_detail WHERE stock_code = %s", (ticker,))
        conn.commit()

        sklearn_models = ['polynomial', 'lasso', 'ridge', 'elasticNet', 'xgboost', 'svm']
        dl_models = ['lstm', 'gru']

        # ── Sklearn 계열 ─────────────────────────────────────────────
        for name in sklearn_models:
            mparams = (params_by_model or {}).get(name, {})
            model, x_scaler, y_scaler, poly = train_sklearn_model(X_train, y_train, name, params=mparams)
            if model:
                trained_models[name] = {'model': model, 'x_scaler': x_scaler, 'y_scaler': y_scaler, 'poly': poly}
                X_full = x_scaler.transform(X_features)
                if poly: X_full = poly.transform(X_full)
                preds_scaled = model.predict(X_full)
                preds = y_scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1))
                mae = mean_absolute_error(y_test, preds[split_index:])
                rmse = np.sqrt(mean_squared_error(y_test, preds[split_index:]))

                log_model_results(cursor, ticker, name, mae, rmse, split_index,
                  stock_data.index[0], stock_data.index[-1],
                  mparams, n_steps=n_steps, variant=variant)
                # 🚨 수정: variant="base" 명시적으로 전달 (app.py의 조회 로직과 통일)
                log_prediction_details(cursor, ticker, name, y_true.index, y_true.values.flatten(), preds.flatten(), n_steps=n_steps, variant=variant)
                last_pred   = float(preds[-1])
                last_close  = float(stock_data['Close'].iloc[-1])

                predict_class = 'UP' if last_pred >= last_close else 'DOWN'

                delta_pct = 0.0 if last_close == 0 else abs(last_pred - last_close) / last_close
                vol = stock_data['Close'].pct_change().rolling(20).std().iloc[-1]
                vol = float(vol) if np.isfinite(vol) and vol > 0 else 0.02  # 기본 변동성 2%
                score = delta_pct / (vol * 2.0)
                confidence = float(np.tanh(score))
                confidence = max(0.01, min(0.99, confidence))

                upsert_prediction_result(
                conn, ticker, date.today(), name,
                last_pred, predict_class, confidence,
                variant=variant, n_steps=n_steps
                )


                nd = next_trading_day(date.today())
                insert_future_prediction(conn, ticker, name, nd, last_pred, run_tag, variant=variant)

        # ── DL 계열 ──────────────────────────────────────────────────
        for name in dl_models:
            dlp = (params_by_model or {}).get(name, {})
            epochs = int(dlp.get('epochs', 10))
            units = int(dlp.get('units', 50))
            batch_size = int(dlp.get('batch_size', 32))

            train_prices = close_prices[:split_index + n_steps]
            model, scaler, _, _ = train_dl_model(train_prices, n_steps, name, epochs=epochs, units=units, batch_size=batch_size)
            if model:
                trained_models[name] = {'model': model, 'scaler': scaler}
                full_scaled = scaler.transform(close_prices.values.reshape(-1, 1))
                X_seq = np.array([full_scaled[i:i + n_steps, 0] for i in range(len(full_scaled) - n_steps)])
                preds = scaler.inverse_transform(model.predict(X_seq.reshape(-1, n_steps, 1), verbose=0))

                mae = mean_absolute_error(y_true.iloc[split_index:], preds[split_index:])
                rmse = np.sqrt(mean_squared_error(y_true.iloc[split_index:], preds[split_index:]))

                dl_logged = {"epochs": epochs, "units": units, "batch_size": batch_size, "n_steps": n_steps}
                log_model_results(cursor, ticker, name, mae, rmse, split_index,
                  stock_data.index[0], stock_data.index[-1],
                  dl_logged, n_steps=n_steps, variant=variant)
                # 🚨 수정: variant="base" 명시적으로 전달 (app.py의 조회 로직과 통일)
                log_prediction_details(cursor, ticker, name, y_true.index, y_true.values.flatten(), preds.flatten(), n_steps=n_steps, variant=variant)

                # 대표 예측(마지막 값) → 결과/익일 저장
                last_pred   = float(preds[-1])
                last_close  = float(stock_data['Close'].iloc[-1])

                predict_class = 'UP' if last_pred >= last_close else 'DOWN'
                delta_pct = 0.0 if last_close == 0 else abs(last_pred - last_close) / last_close
                vol = stock_data['Close'].pct_change().rolling(20).std().iloc[-1]
                vol = float(vol) if np.isfinite(vol) and vol > 0 else 0.02
                score = delta_pct / (vol * 2.0)
                confidence = float(np.tanh(score))
                confidence = max(0.01, min(0.99, confidence))

                upsert_prediction_result(
                conn, ticker, date.today(), name,
                last_pred, predict_class, confidence,
                variant=variant, n_steps=n_steps
                )

                nd = next_trading_day(date.today())
                insert_future_prediction(conn, ticker, name, nd, last_pred, run_tag, variant=variant)                                  

        conn.commit()
        logger.info(f"'{ticker}'에 대한 모든 모델 학습 및 DB 저장을 완료했습니다.")
    except Exception as e:
        logger.error(f"train_and_predict_all_models 함수에서 예외 발생: {e}", exc_info=True)
        if conn and conn.is_connected():
            conn.rollback()
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()

def log_model_results(cursor, code, name, mae, rmse, test_idx, start, end, params, n_steps, variant="base"):
    query = """
        INSERT INTO model_comparison_log
        (stock_code, model_name, train_start, train_end, mae, rmse, test_start_index, n_steps, parameters, variant)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (code, name, start, end, mae, rmse, test_idx, n_steps, json.dumps(params), variant))

def log_prediction_details(cursor, code, name, dates, actuals, preds, n_steps, variant="base"):
    query = """
        INSERT INTO model_prediction_detail
        (stock_code, model_name, target_date, actual_price, predicted_price, n_steps, variant)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    data = []
    for dt, actual, pred in zip(dates, actuals, preds):
        if actual is not None and pred is not None and np.isfinite(actual) and np.isfinite(pred):
            data.append((code, name, dt, float(actual), float(pred), int(n_steps), variant))
    if data:
        cursor.executemany(query, data)

def train_sklearn_model(X, y, model_name, params=None):
    params = params or {}

    # 스케일러: 원시 코드와 맞추려면 아래 두 줄로 교체 (원하면 유지 가능)
    from sklearn.preprocessing import MinMaxScaler
    x_scaler = MinMaxScaler().fit(X)     # ← 원시처럼 통일
    y_scaler = MinMaxScaler().fit(y)

    Xs = x_scaler.transform(X)
    ys = y_scaler.transform(y)
    model, poly = None, None

    if model_name == 'polynomial':
        deg = int(params.get('degree', 2))
        poly = PolynomialFeatures(degree=deg).fit(Xs)
        # 원시처럼 Ridge 사용 + alpha 주입
        ridge_alpha = float(params.get('ridge_alpha', 1.0))
        model = Ridge(alpha=ridge_alpha).fit(poly.transform(Xs), ys)
    else:
        # 각 모델별 하이퍼파라미터 주입
        if model_name == 'lasso':
            model = Lasso(alpha=float(params.get('alpha', 0.01)), max_iter=int(params.get('max_iter', 10000)))
        elif model_name == 'ridge':
            model = Ridge(alpha=float(params.get('alpha', 1.0)))
        elif model_name == 'elasticNet':
            model = ElasticNet(alpha=float(params.get('alpha', 0.01)),
                               l1_ratio=float(params.get('l1_ratio', 0.5)),
                               max_iter=int(params.get('max_iter', 10000)))
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=float(params.get('learning_rate', params.get('eta', 0.01))),
                max_depth=int(params.get('max_depth', 3)),
                n_estimators=int(params.get('n_estimators', 300)),
                subsample=float(params.get('subsample', 0.7)),
                colsample_bytree=float(params.get('colsample_bytree', 0.7)),
                random_state=int(params.get('random_state', 42)),
            )
        elif model_name == 'svm':
            model = SVR(
                C=float(params.get('C', 1.0)),
                gamma=params.get('gamma', 'auto'),   # 원시와 동일
                kernel=params.get('kernel', 'rbf'),
            )
        else:
            return None, x_scaler, y_scaler, poly

        model.fit(Xs, ys.ravel())

    return model, x_scaler, y_scaler, poly

def train_dl_model(data_series, n_steps, model_name, epochs=10, units=50, batch_size=32):
    tf.keras.backend.clear_session()
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_series.values.reshape(-1, 1))
    scaled_data = scaler.transform(data_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:(i + n_steps), 0])
        y.append(scaled_data[i + n_steps, 0])
    if not X:
        return None, None, None, None

    X_train = np.array(X).reshape(-1, n_steps, 1)
    y_train = np.array(y)

    rnn_layer = tf.keras.layers.LSTM if model_name == 'lstm' else tf.keras.layers.GRU
    model = tf.keras.Sequential([
        rnn_layer(units, input_shape=(n_steps, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler, X_train, y_train


#25-10-17추가(익일 예측 비교 및 파라미터값 +,- 비교)
import os
VARIANT = os.getenv("VARIANT", "base")

def get_conn():
    return mysql.connect(**DB_CONFIG)

def next_trading_day(d):
    # 한국 휴일/주말 처리 간단화(주말만 스킵). 필요시 휴일 테이블로 보강.
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:  # 5,6 = 토/일
        nd += timedelta(days=1)
    return nd

def upsert_stock_info_if_missing(conn, stock_code, stock_name=None, industry=None, market_type=None):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM stock_info WHERE stock_code=%s", (stock_code,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO stock_info (stock_code, stock_name, industry, market_type) VALUES (%s,%s,%s,%s)",
            (stock_code, stock_name or stock_code, industry or "N/A", market_type or "N/A"),
        )
        conn.commit()
    cur.close()

def insert_model_logs(conn, stock_code, model_name, train_start, train_end, mae, rmse, params_json, model_path=None, variant="base"):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_comparison_log
        (stock_code, model_name, train_start, train_end, mae, rmse, test_start_index, parameters, model_path, variant)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (stock_code, model_name, train_start, train_end, mae, rmse, None, json.dumps(params_json), model_path, variant))
    conn.commit()
    cur.close()

def insert_prediction_detail(conn, stock_code, model_name, target_date, actual_price, predicted_price, variant="base"):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_prediction_detail
        (stock_code, model_name, target_date, actual_price, predicted_price, variant)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (stock_code, model_name, target_date, actual_price, predicted_price, variant))
    conn.commit()
    cur.close()

def upsert_prediction_result(conn, stock_code, target_date, model_name, predict_value, predict_class, confidence, variant="base", n_steps=None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_result
        (stock_code, date, model_name, predict_value, predict_class, confidence, variant, n_steps)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          predict_value=VALUES(predict_value),
          predict_class=VALUES(predict_class),
          confidence=VALUES(confidence),
          n_steps=VALUES(n_steps)
    """, (stock_code, target_date, model_name, predict_value, predict_class, confidence, variant, n_steps))
    conn.commit()
    cur.close()

def insert_future_prediction(conn, stock_code, model_name, prediction_date, predicted_price, run_tag=None, variant="base"):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO future_predictions
        (stock_code, model_name, prediction_date, predicted_price, variant, run_tag)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          predicted_price=VALUES(predicted_price),
          run_tag=VALUES(run_tag)
    """, (stock_code, model_name, prediction_date, predicted_price, variant, run_tag))
    conn.commit()
    cur.close()

# === 아래 두 함수는 네 모델 로직에 맞게 구현만 바꾸면 됨 ===
def train_and_predict_one_model(stock_code, model_name, params):
    """
    반환: (train_start, train_end, mae, rmse, today_pred_price, today_conf, cls_label)
    - train_start/train_end : 학습에 실제로 사용된 기간의 시작/끝 날짜
    - mae/rmse              : 테스트셋 성능
    - today_pred_price      : 가장 최근 윈도우에 대한 예측값(= '오늘'에 해당)
    - today_conf            : 분류는 최대 확률, 회귀는 0.0(필요시 정의)
    - cls_label             : 분류는 예측 라벨, 회귀는 예측값 vs 마지막 종가의 up/down
    """
    # ===== 1) 데이터 로드 & 기본 세팅 =====
    n_steps   = int(params.get("n_steps", 60))
    test_size = float(params.get("test_size", 0.2))
    stock_data = get_and_store_stock_data(stock_code, days_back=int(params.get("historical_days", 1095)))
    if stock_data is None or stock_data.empty:
        raise RuntimeError(f"{stock_code}: 학습에 사용할 데이터가 없습니다.")

    # Close/Volume 기준으로 랙 특성 구성(네 all-in-one 로직과 동일)
    df_features = stock_data[['Close', 'Volume']].copy()
    close = df_features['Close']
    lag_list = [close.shift(i).rename(f'lag_{i}') for i in range(1, n_steps + 1)]
    lag_block = pd.concat(lag_list, axis=1)
    df_features = pd.concat([df_features, lag_block], axis=1)

    df_features = df_features.copy()

    df_features = df_features.dropna()
    if df_features.empty:
        raise RuntimeError(f"{stock_code}: 전처리 후 유효 표본이 없습니다. n_steps={n_steps}가 너무 클 수 있습니다.")

    y_true      = df_features[['Close']]
    X_features  = df_features.drop('Close', axis=1)
    split_index = int(len(X_features) * (1 - test_size))
    X_train, X_test = X_features[:split_index], X_features[split_index:]
    y_train, y_test = y_true[:split_index], y_true[split_index:]

    # 학습 구간 날짜(네 로그 함수와 기준 통일)
    train_start = pd.to_datetime(stock_data.index[0]).date()
    # split_index는 테스트 시작이므로, 학습 마지막은 그 직전 샘플의 '원본' 날짜와 거의 같음
    train_end   = pd.to_datetime(stock_data.index[max(0, split_index - 1)]).date()

    # ===== 2) 모델 학습 & 전체 구간 예측 =====
    sklearn_models = {'polynomial', 'lasso', 'ridge', 'elasticNet', 'xgboost', 'svm'}
    dl_models      = {'lstm', 'gru'}

    if model_name in sklearn_models:
        mparams = params.copy()
        model, x_scaler, y_scaler, poly = train_sklearn_model(X_train, y_train, model_name, params=mparams)
        if model is None:
            raise RuntimeError(f"{stock_code}/{model_name}: 모델 생성 실패")

        X_full = x_scaler.transform(X_features)
        if poly:
            X_full = poly.transform(X_full)
        preds_scaled = model.predict(X_full)
        preds = y_scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()

    elif model_name in dl_models:
        dlp       = params.copy()
        epochs    = int(dlp.get('epochs', 10))
        units     = int(dlp.get('units', 50))
        batch_sz  = int(dlp.get('batch_size', 32))

        close_prices = stock_data['Close']
        # 네 기존 로직처럼: 학습은 (학습종료지점 + n_steps)까지만 사용
        train_prices = close_prices[:split_index + n_steps]
        model, scaler, _, _ = train_dl_model(train_prices, n_steps, model_name,
                                             epochs=epochs, units=units, batch_size=batch_sz)
        if model is None:
            raise RuntimeError(f"{stock_code}/{model_name}: DL 모델 학습 실패")

        full_scaled = scaler.transform(close_prices.values.reshape(-1, 1))
        X_seq = np.array([full_scaled[i:i + n_steps, 0] for i in range(len(full_scaled) - n_steps)])
        # 전체 구간 예측(스케일 역변환)
        preds = scaler.inverse_transform(model.predict(X_seq.reshape(-1, n_steps, 1), verbose=0)).ravel()
        # 특성 랙 때문에 df_features 기준과 길이가 맞도록 패딩
        # df_features는 n_steps 이후부터 시작하므로 preds의 길이와 정렬이 동일
    else:
        raise ValueError(f"알 수 없는 모델명: {model_name}")

    # ===== 3) 성능/오늘 예측/라벨 =====
    # df_features는 n_steps 이후 시점부터, preds도 같은 기준이므로 바로 split 사용 가능
    mae  = float(mean_absolute_error(y_test.values.ravel(), preds[split_index:]))
    rmse = float(np.sqrt(mean_squared_error(y_test.values.ravel(), preds[split_index:])))

    today_pred_price = float(preds[-1])

    # 분류면 확률, 회귀면 0.0 + up/down
    today_conf = 0.0
    cls_label  = "up" if today_pred_price >= float(stock_data['Close'].iloc[-1]) else "down"

    return train_start, train_end, mae, rmse, today_pred_price, today_conf, cls_label

def run_for_ticker_list(tickers, models, run_tag=None, variant="base"):
    """
    tickers: ["005930.KS", ...]
    models:  {"gru": {...}, "lstm": {...}, "xgboost": {...}, ...}
    variant: "minus" | "base" | "plus"
    """
    ns = models.get("gru", {}).get("n_steps") or models.get("lstm", {}).get("n_steps") or 60

    for tk in tickers:
        print(f"[RUN] {tk} (models={list(models.keys())}, variant={variant}, tag={run_tag})")
        try:
            _ = train_and_predict_all_models(
                ticker=tk,
                historical_days=1095,
                n_steps=ns,
                test_size=0.2,
                params_by_model=models,
                run_tag=run_tag,
                variant=variant,     
            )
            print(f"[DONE] {tk} variant={variant}")

            inserted = compute_and_insert_alerts_direct(
                conn=get_db_connection(),       # 네가 쓰는 DB 커넥터
                target_date=None,               # None이면 최신 예측일
                direction=os.getenv("ALERTS_DIRECTION", "both"),
                require_all=os.getenv("ALERTS_REQUIRE_ALL", "true").lower() == "true",
                min_confidence=(float(os.getenv("ALERTS_MIN_CONF", "0.7")) if os.getenv("ALERTS_MIN_CONF") else None),
                min_models=(int(os.getenv("ALERTS_MIN_MODELS")) if os.getenv("ALERTS_MIN_MODELS") else None),
                alert_prefix=os.getenv("ALERTS_PREFIX", "consensus")
            )
            logger.info(f"alerts.direct inserted={inserted}")

        except Exception as e:
            print(f"[ERROR] {tk} variant={variant}: {e!r}")

# === 추가: 알림 직접 생성 함수 ===
def compute_and_insert_alerts_direct(conn, target_date=None,
                                     direction="both",
                                     require_all=True,
                                     min_confidence=None,
                                     min_models=None,
                                     alert_prefix="consensus"):
    """/api/alerts/run 대신 DB에 직접 합의 알림을 적재한다."""
    if conn is None:
        logger.warning("알림 생성 실패: DB 커넥션 없음")
        return 0

    with conn.cursor() as cur:
        # 날짜 미지정이면 prediction_result 최신일
        if target_date is None:
            cur.execute("SELECT MAX(date) AS d FROM prediction_result")
            row = cur.fetchone()
            target_date = row[0]
            if not target_date:
                logger.warning("prediction_result가 비어있어 알림 생성 스킵")
                return 0

        # confidence 조건
        conf_sql = "AND (confidence IS NULL OR confidence >= %s)" if (min_confidence is not None) else ""
        conf_param = [float(min_confidence)] if (min_confidence is not None) else []

        # (stock_code, date)별 up/down 카운트 집계
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

        # 조건식
        up_cond_sql = "g.n_up = g.n_models" if require_all else "g.n_up >= COALESCE(%s, CEIL(g.n_models/2))"
        dn_cond_sql = "g.n_down = g.n_models" if require_all else "g.n_down >= COALESCE(%s, CEIL(g.n_models/2))"

        inserted_total = 0

        def insert_by_direction(is_up: bool):
            nonlocal inserted_total
            alert_type   = f"{alert_prefix}_{'up' if is_up else 'down'}"
            dir_label    = "상승" if is_up else "하락"
            consensus_tag = " 합의" if alert_prefix == "consensus" else ""
            conf_tag     = f" (conf≥{min_confidence})" if min_confidence is not None else ""
            if not require_all:
                models_tag = f" (≥{int(min_models)} models)" if (min_models is not None) else " (≥과반)"
            else:
                models_tag = ""
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
                alert_type,
                message_prefix,
                # group_sql 파라미터
            ] + conf_param + conf_param + [target_date] + [
                # LEFT JOIN alert_type, pr.date
                alert_type, target_date
            ]
            if not require_all:
                params += [min_models]

            cur.execute(sql, params)
            inserted_total += cur.rowcount

        if direction in ("up", "both"):
            insert_by_direction(True)
        if direction in ("down", "both"):
            insert_by_direction(False)

        conn.commit()
        return inserted_total




ALERTS_BASE_URL = os.getenv("ALERTS_BASE_URL", "http://localhost:5000")  # Cloud Run URL로 교체 가능
ALERTS_ENDPOINT = f"{ALERTS_BASE_URL.rstrip('/')}/api/alerts/run"



def trigger_alerts_run(direction="up", require_all=True, min_confidence=None, min_models=None, alert_prefix="consensus"):
    """학습 완료 후 알림 재계산을 트리거한다."""
    payload = {
        "direction": direction,          # "up" | "down" | "both"
        "require_all": require_all,      # 전(全)모델 일치
        "min_confidence": min_confidence,# e.g., 0.7 (없으면 None)
        "min_models": min_models,        # require_all=False일 때만 사용
        "alert_prefix": alert_prefix     # "consensus" | "model"
        # "date"는 생략 → 백엔드가 MAX(date) 사용
    }
    try:
        r = requests.post(ALERTS_ENDPOINT, json=payload, timeout=15)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        logger.warning(f"/api/alerts/run 호출 실패: {e}")
        return False, {"error": str(e)}

