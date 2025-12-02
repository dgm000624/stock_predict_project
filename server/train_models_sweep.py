import os, argparse, time
from datetime import datetime
from pathlib import Path
import importlib
import json

# ↓ 텐서플로우 로그 줄이기(선택)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---- import 경로 안전하게 고정 ----
SERVER_DIR = Path(__file__).parent
PROJECT_DIR = SERVER_DIR.parent.parent  # c:\Project_Team_A
import sys
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import train_models
importlib.reload(train_models)  # 캐시 무시하고 최신 로드

# ======================
# 1) 파라미터 프리셋
# ======================
# train_models.py에서 사용하는 모델 이름에 맞춰 키를 잡아 주세요.
# (추정: 'gru','lstm','xgboost','ridge','lasso','elasticNet','svm','polynomial')
# - 쓰지 않는 키는 내부에서 무시돼도 안전
# - 새로운 모델을 추가하면 여기에 항목만 추가하면 됨

# 공통 기본값(variant와 무관하게 시작점)
PARAM_BASE = {
    "gru":  {"n_steps": 180, "epochs": 14},   
    "lstm": {"n_steps": 180, "epochs": 14},   

    "xgboost": {
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    "ridge":      {"alpha": 0.5,   "random_state": 42},
    "lasso":      {"alpha": 0.0005, "random_state": 42, "max_iter": 10000},
    "elasticNet": {"alpha": 0.0005, "l1_ratio": 0.4, "random_state": 42, "max_iter": 10000},
    "svm":        {"C": 0.7, "gamma": "scale", "kernel": "rbf", "random_state": 42},
    "polynomial": {
        "degree": 2,
        "ridge_alpha": 0.7,
        "include_bias": True,
        "interaction_only": False,
    },
}

PARAM_OVERLAY = {
    # BASE보다 가볍게(단순/규제↑/복잡도↓)
    "minus": {
        "gru":  {"n_steps": 180, "epochs": 10},
        "lstm": {"n_steps": 180, "epochs": 10},

        "xgboost": {
            "max_depth": 2,
            "n_estimators": 150,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        },
        "ridge": {"alpha": 0.3, "random_state": 42},
        "lasso": {"alpha": 0.0003, "random_state": 42, "max_iter": 10000},
        "elasticNet": {"alpha": 0.0003, "l1_ratio": 0.35, "random_state": 42, "max_iter": 10000},
        "svm": {"C": 0.5, "gamma": "scale", "kernel": "rbf", "random_state": 42},
        "polynomial": {
            "degree": 2,
            "ridge_alpha": 0.5,
            "include_bias": True,
            "interaction_only": False,
        },
    },

    "base": {},  # 위 PARAM_BASE 그대로 사용

    # BASE보다 공격적으로(복잡도↑/규제↓)
    "plus": {
        "gru":  {"n_steps": 180, "epochs": 18},
        "lstm": {"n_steps": 180, "epochs": 18},

        # 기존 기본안(옛 base)에 가깝게 상향
        "xgboost": {
            "max_depth": 4,
            "n_estimators": 300,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        },
        "ridge": {"alpha": 1.0, "random_state": 42},
        "lasso": {"alpha": 0.001, "random_state": 42, "max_iter": 10000},
        "elasticNet": {"alpha": 0.001, "l1_ratio": 0.5, "random_state": 42, "max_iter": 10000},
        "svm": {"C": 1.0, "gamma": "scale", "kernel": "rbf", "random_state": 42},
        "polynomial": {
            "degree": 3,
            "ridge_alpha": 1.0,
            "include_bias": True,
            "interaction_only": False,
        },
    },
}

#VARIANTS = ["minus", "base", "plus"]
VARIANTS = ["base"]


def deep_merge(a: dict, b: dict) -> dict:
    """얕은 dict 덮어쓰기(중첩 dict면 내부까지 병합)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            inner = dict(out[k])
            inner.update(v)
            out[k] = inner
        else:
            out[k] = v
    return out


def build_params_for_variant(variant: str, only_models=None):
    """variant별로 모델 파라미터 세트를 생성(기본값 + overlay)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    base = PARAM_BASE
    overlay = PARAM_OVERLAY.get(variant, {})
    merged = {}
    model_keys = only_models or base.keys()
    for m in model_keys:
        base_m = base.get(m, {})
        over_m = overlay.get(m, {})
        merged[m] = deep_merge(base_m, over_m)
    return merged


# ======================
# 2) 유틸: 티커/모델 로딩
# ======================
def load_tickers(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    out, seen = [], set()
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for chunk in line.split(",") for x in chunk.split() if x.strip()]
            for tk in parts:
                if tk not in seen:
                    seen.add(tk); out.append(tk)
    if not out:
        raise ValueError("No tickers parsed from file")
    return out


# ======================
# 3) 스윕 실행
# ======================
def run_sweep(tickers, only_variants=None, exclude_variants=None, only_models=None, run_tag=None):
    variants = [v for v in VARIANTS
                if (not only_variants or v in set(only_variants))
                and (not exclude_variants or v not in set(exclude_variants))]
    base_variant_env = os.environ.get("VARIANT")

    try:
        for variant in variants:
             # os.environ["VARIANT"] = variant  # ← 삭제
            params_by_model = build_params_for_variant(variant, only_models=only_models)

            tag = run_tag or f"{datetime.now():%Y%m%d}-close"
            print(f"[SWEEP] variant={variant} tickers={len(tickers)} models={list(params_by_model.keys())} run_tag={tag}")

            t0 = time.time()
            # ★ 중요: variant를 인자로 명시 전달
            train_models.run_for_ticker_list(tickers, params_by_model, run_tag=tag, variant=variant)
            print(f"[DONE]  variant={variant} elapsed={time.time()-t0:.1f}s")
    finally:
        if base_variant_env is None:
            os.environ.pop("VARIANT", None)
        else:
            os.environ["VARIANT"] = base_variant_env


# ======================
# 4) CLI
# ======================
def parse_args():
    p = argparse.ArgumentParser(description="Parameter sweep over multiple models and variants.")
    p.add_argument("-f", "--tickers-file", default=str(SERVER_DIR / "tickers.txt"),
                   help="Path to tickers list file (default: ./tickers.txt)")
    p.add_argument("--only-variants", nargs="*", choices=VARIANTS,
                   help="Run only selected variants (e.g., --only-variants base plus)")
    p.add_argument("--exclude-variants", nargs="*", choices=VARIANTS,
                   help="Exclude variants (e.g., --exclude-variants minus)")
    p.add_argument("--only-models", nargs="*", default=None,
                   help="Limit to selected models (e.g., --only-models gru lstm xgboost)")
    p.add_argument("--run-tag", default=None, help="Optional run tag (default: YYYYMMDD-close)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tickers = load_tickers(args.tickers_file)
    run_sweep(
        tickers,    
        only_variants=args.only_variants,
        exclude_variants=args.exclude_variants,
        only_models=args.only_models,
        run_tag=args.run_tag
    )
