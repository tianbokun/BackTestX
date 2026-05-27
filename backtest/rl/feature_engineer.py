import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None

# ── Feature group definitions ──
FEATURE_GROUPS = {
    "trend": {
        "label": "趋势指标 (MACD+ADX/DI±)",
        "columns": ["MACD_DIF", "MACD_DEA", "MACD_HIST", "ADX", "DI_PLUS", "DI_MINUS"],
        "raw_columns": [],
        "default": True,
        "help": "MACD 三线 + ADX 方向分量",
    },
    "price_position": {
        "label": "价格相对位置",
        "columns": ["close_ma20_ratio", "percent_b"],
        "raw_columns": ["percent_b"],
        "default": True,
        "help": "收盘价/MA20 偏离度 + Bollinger %B (0~1)",
    },
    "momentum": {
        "label": "动量 (RSI14)",
        "columns": ["RSI14"],
        "raw_columns": ["RSI14"],
        "default": True,
        "help": "RSI14 原始值 (0-100)",
    },
    "volatility": {
        "label": "波动率 (ATR)",
        "columns": ["ATR"],
        "raw_columns": [],
        "default": True,
        "help": "平均真实波幅",
    },
    "ma_cross": {
        "label": "均线交叉 (MA5/10/30/120)",
        "columns": ["MA5", "MA10", "MA30", "MA120"],
        "raw_columns": [],
        "default": False,
        "help": "传统均线系统（与 MACD 高度相关）",
    },
    "kdj": {
        "label": "KDJ 随机指标",
        "columns": ["KDJ_K", "KDJ_D", "KDJ_J"],
        "raw_columns": ["KDJ_K", "KDJ_D", "KDJ_J"],
        "default": False,
        "help": "超买超卖信号（与 RSI 功能重叠）",
    },
    "rsi_detail": {
        "label": "多周期 RSI (6/12/24)",
        "columns": ["RSI6", "RSI12", "RSI24"],
        "raw_columns": ["RSI6", "RSI12", "RSI24"],
        "default": False,
        "help": "多周期 RSI 叠加",
    },
    "volume": {
        "label": "量价指标 (OBV+成交额)",
        "columns": ["OBV", "成交额"],
        "raw_columns": [],
        "default": False,
        "help": "能量潮 OBV + 成交额",
    },
}

DEFAULT_FEATURE_GROUPS = [k for k, v in FEATURE_GROUPS.items() if v.get("default", False)]


def get_selected_columns(feature_groups: list[str]) -> tuple:
    all_cols = ["close"]
    raw_cols = []
    for key in feature_groups:
        grp = FEATURE_GROUPS.get(key)
        if grp is None:
            continue
        all_cols.extend(grp["columns"])
        raw_cols.extend(grp.get("raw_columns", []))
    seen = set()
    dedup = []
    for c in all_cols:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    raw_cols = [c for c in raw_cols if c in seen]
    return dedup, raw_cols


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ohclv = df.copy()

    def _col(names):
        for n in names:
            if n in ohclv.columns:
                return ohclv[n].values.astype(float)
        return np.ones(len(ohclv))

    close = _col(["收盘", "收盘价", "close"])
    high = _col(["最高", "最高价", "high"])
    low = _col(["最低", "最低价", "low"])
    open_p = _col(["开盘", "开盘价", "open"])
    volume = _col(["成交量", "volume"])

    out = pd.DataFrame(index=ohclv.index)
    out["close"] = close
    out["high"] = high
    out["low"] = low
    out["open"] = open_p
    out["volume"] = volume

    # ── helpers ──
    def _ma(x, period):
        return pd.Series(x).rolling(period, min_periods=1).mean().values

    def _ema(x, period):
        return pd.Series(x).ewm(span=period, min_periods=1, adjust=False).mean().values

    def _bbands(x, period=20, nbdev=2):
        s = pd.Series(x)
        ma = s.rolling(period, min_periods=1).mean()
        std = s.rolling(period, min_periods=1).std(ddof=0)
        return ma.values, (ma + nbdev * std).values, (ma - nbdev * std).values

    def _atr(h, l, c, period=14):
        s_h, s_l, s_c = pd.Series(h), pd.Series(l), pd.Series(c)
        tr = pd.concat([s_h - s_l, (s_h - s_c.shift()).abs(), (s_l - s_c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean().values

    def _adx_di(h, l, c, period=14):
        s_h, s_l, s_c = pd.Series(h), pd.Series(l), pd.Series(c)
        up = s_h - s_h.shift()
        down = s_l.shift() - s_l
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr = pd.concat([s_h - s_l, (s_h - s_c.shift()).abs(), (s_l - s_c.shift()).abs()], axis=1).max(axis=1)
        tr_rolling = tr.rolling(period, min_periods=1).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period, min_periods=1).mean() / tr_rolling.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm).rolling(period, min_periods=1).mean() / tr_rolling.replace(0, np.nan)
        dx = abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.rolling(period, min_periods=1).mean()
        return adx.values, plus_di.values, minus_di.values

    def _cci(h, l, c, period=20):
        tp = (h + l + c) / 3
        s_tp = pd.Series(tp)
        ma = s_tp.rolling(period, min_periods=1).mean().values
        md = s_tp.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
        md = np.where(md == 0, np.nan, md)
        return (tp - ma) / (0.015 * md)

    def _kdj(h, l, c, period=9):
        s_h, s_l, s_c = pd.Series(h), pd.Series(l), pd.Series(c)
        hh = s_h.rolling(period, min_periods=1).max()
        ll = s_l.rolling(period, min_periods=1).min()
        rsv = (c - ll) / (hh - ll).replace(0, np.nan) * 100
        k = rsv.ewm(span=3, min_periods=1, adjust=False).mean().values
        d = pd.Series(k).ewm(span=3, min_periods=1, adjust=False).mean().values
        j = 3 * k - 2 * d
        return k, d, j

    def _rsi(x, period=14):
        s = pd.Series(x)
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).values

    def _macd(close, fast=12, slow=26, signal=9):
        s = pd.Series(close)
        ema_f = s.ewm(span=fast, min_periods=1, adjust=False).mean()
        ema_s = s.ewm(span=slow, min_periods=1, adjust=False).mean()
        dif = ema_f - ema_s
        dea = dif.ewm(span=signal, min_periods=1, adjust=False).mean()
        hist = 2 * (dif - dea)
        return dif.values, dea.values, hist.values

    def _obv(close, volume):
        direction = np.sign(np.diff(close, prepend=close[0]))
        direction[0] = 0
        return np.cumsum(direction * volume)

    # ── Moving averages (legacy) ──
    out["MA5"] = _ma(close, 5)
    out["MA10"] = _ma(close, 10)
    out["MA30"] = _ma(close, 30)
    out["MA120"] = _ma(close, 120)
    out["EMA30"] = _ema(close, 30)

    # ── Price position ──
    ma20 = _ma(close, 20)
    out["close_ma20_ratio"] = close / np.maximum(ma20, 1e-10) - 1.0

    # ── Bollinger Bands ──
    bb_mid, bb_up, bb_low = _bbands(close)
    out["BB_mid"] = bb_mid
    out["BB_up"] = bb_up
    out["BB_low"] = bb_low
    out["percent_b"] = np.clip((close - bb_low) / np.maximum(bb_up - bb_low, 1e-10), 0.0, 1.0)

    # ── Trend ──
    out["ADX_raw"], out["DI_PLUS"], out["DI_MINUS"] = _adx_di(high, low, close)
    macd_dif, macd_dea, macd_hist = _macd(close)
    out["MACD_DIF"], out["MACD_DEA"], out["MACD_HIST"] = macd_dif, macd_dea, macd_hist

    # ── Volatility ──
    out["ATR"] = _atr(high, low, close)

    # ── Momentum ──
    out["RSI14"] = _rsi(close, 14)
    out["CCI"] = _cci(high, low, close)

    # ── KDJ ──
    k, d, j = _kdj(high, low, close)
    out["KDJ_K"], out["KDJ_D"], out["KDJ_J"] = k, d, j

    # ── Multi-period RSI (legacy) ──
    out["RSI6"] = _rsi(close, 6)
    out["RSI12"] = _rsi(close, 12)
    out["RSI24"] = _rsi(close, 24)

    # ── Volume ──
    out["OBV"] = _obv(close, volume)

    # ── Premium rate (ETF/LOF only) ──
    if "溢价率" in ohclv.columns:
        out["溢价率"] = ohclv["溢价率"].values.astype(float)
    else:
        out["溢价率"] = np.zeros(len(ohclv))

    return out


def normalize_indicators(df: pd.DataFrame, raw_columns: list[str] = None) -> pd.DataFrame:
    df = df.ffill().fillna(0)
    if raw_columns is None:
        raw_columns = []
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].values.astype(np.float64)
        if col in raw_columns:
            result[col] = s.astype(np.float32)
            continue
        n = len(s)
        norm = np.zeros(n, dtype=np.float32)
        cumsum = 0.0
        cumsum_sq = 0.0
        for t in range(n):
            cumsum += s[t]
            cumsum_sq += s[t] * s[t]
            count = t + 1
            mean = cumsum / count
            var = max(cumsum_sq / count - mean * mean, 1e-16)
            std = np.sqrt(var)
            if std < 1e-8:
                norm[t] = 0.0
            else:
                norm[t] = np.clip((s[t] - mean) / std, -5.0, 5.0)
        result[col] = norm
    return result


def get_state_vector(
    df_indicators: pd.DataFrame,
    t: int,
    system_version: str = "1.0",
    feature_groups: list[str] = None,
    svm_signal: np.ndarray = None,
    xgb_signal: np.ndarray = None,
) -> np.ndarray:
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    if system_version == "basic":
        return np.array([df_indicators["close"].iloc[t]], dtype=np.float32)
    cols, _ = get_selected_columns(feature_groups)
    vec = df_indicators[cols].iloc[t].values.astype(np.float32)
    if np.any(np.isnan(vec)):
        vec = np.nan_to_num(vec, nan=0.0)
    if system_version == "2.0" and svm_signal is not None and xgb_signal is not None:
        sig = np.array([svm_signal[t], xgb_signal[t]], dtype=np.float32)
        vec = np.concatenate([vec, sig])
    return vec


def get_state_dim(system_version: str = "1.0", feature_groups: list[str] = None) -> int:
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    if system_version == "basic":
        return 1
    cols, _ = get_selected_columns(feature_groups)
    dim = len(cols)
    if system_version == "2.0":
        dim += 2
    return dim


def compute_svm_xgb_signals(df: pd.DataFrame, window: int = 30) -> tuple:
    from sklearn.svm import LinearSVC
    import xgboost as xgb

    def _col(names):
        for n in names:
            if n in df.columns:
                return df[n].values.astype(float)
        return np.ones(len(df))

    close = _col(["收盘", "收盘价", "close"])
    labels = np.where(np.diff(close, prepend=close[0]) > 0, 1, -1)

    cols = []
    for names in [["开盘价", "开盘", "open"], ["收盘价", "收盘", "close"],
                  ["最高价", "最高", "high"], ["最低价", "最低", "low"],
                  ["成交量", "volume"]]:
        for n in names:
            if n in df.columns:
                cols.append(n)
                break
    if not cols:
        return np.zeros(len(close)), np.zeros(len(close))

    data = df[cols].values.astype(float)
    data = np.nan_to_num(data, nan=0.0)

    svm_sig = np.zeros(len(close))
    xgb_sig = np.zeros(len(close))

    for t in range(window, len(close)):
        X = data[t - window:t]
        y = labels[t - window:t]
        if len(np.unique(y)) < 2:
            continue

        weights = np.exp(0.08 * np.arange(len(y)))
        try:
            svm = LinearSVC(C=1, loss="squared_hinge", max_iter=2000, random_state=42)
            svm.fit(X, y)
            svm_sig[t] = svm.predict(data[t].reshape(1, -1))[0]
        except Exception:
            svm_sig[t] = 0

        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, verbosity=0,
                use_label_encoder=False, random_state=42,
            )
            xgb_model.fit(X, y, sample_weight=weights)
            xgb_sig[t] = xgb_model.predict(data[t].reshape(1, -1))[0]
        except Exception:
            xgb_sig[t] = 0

    return svm_sig, xgb_sig
