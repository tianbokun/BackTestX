import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None


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

    n = len(close)
    out = pd.DataFrame(index=ohclv.index)

    out["close"] = close
    out["high"] = high
    out["low"] = low
    out["open"] = open_p
    out["volume"] = volume

    def _ma(x, period):
        s = pd.Series(x).rolling(period, min_periods=1).mean()
        return s.values

    def _ema(x, period):
        s = pd.Series(x).ewm(span=period, min_periods=1, adjust=False).mean()
        return s.values

    def _bbands(x, period=20, nbdev=2):
        s = pd.Series(x)
        ma = s.rolling(period, min_periods=1).mean()
        std = s.rolling(period, min_periods=1).std(ddof=0)
        return ma.values, (ma + nbdev * std).values, (ma - nbdev * std).values

    def _atr(h, l, c, period=14):
        s_h, s_l, s_c = pd.Series(h), pd.Series(l), pd.Series(c)
        tr = pd.concat([s_h - s_l, (s_h - s_c.shift()).abs(), (s_l - s_c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean().values

    def _adx(h, l, c, period=14):
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
        return dx.rolling(period, min_periods=1).mean().values

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

    out["MA5"] = _ma(close, 5)
    out["MA10"] = _ma(close, 10)
    out["MA30"] = _ma(close, 30)
    out["MA120"] = _ma(close, 120)
    out["EMA30"] = _ema(close, 30)

    bb_mid, bb_up, bb_low = _bbands(close)
    out["BB_mid"] = bb_mid
    out["BB_up"] = bb_up
    out["BB_low"] = bb_low

    out["ATR"] = _atr(high, low, close)
    out["ADX"] = _adx(high, low, close)
    out["CCI"] = _cci(high, low, close)

    k, d, j = _kdj(high, low, close)
    out["KDJ_K"], out["KDJ_D"], out["KDJ_J"] = k, d, j

    out["RSI6"] = _rsi(close, 6)
    out["RSI12"] = _rsi(close, 12)
    out["RSI24"] = _rsi(close, 24)

    if "溢价率" in ohclv.columns:
        out["溢价率"] = ohclv["溢价率"].values.astype(float)
    else:
        out["溢价率"] = np.zeros(len(ohclv))

    return out


def get_state_vector(
    df_indicators: pd.DataFrame,
    t: int,
    system_version: str = "1.0",
    svm_signal: np.ndarray = None,
    xgb_signal: np.ndarray = None,
) -> np.ndarray:
    cols = ["close", "MA5", "MA10", "MA30", "MA120", "EMA30",
            "BB_mid", "ATR", "ADX", "CCI",
            "KDJ_K", "KDJ_D", "KDJ_J",
            "RSI6", "RSI12", "RSI24", "volume",
            "溢价率"]
    if system_version == "basic":
        return np.array([df_indicators["close"].iloc[t]], dtype=np.float32)
    vec = df_indicators[cols].iloc[t].values.astype(np.float32)
    if np.any(np.isnan(vec)):
        vec = np.nan_to_num(vec, nan=0.0)
    if system_version == "2.0" and svm_signal is not None and xgb_signal is not None:
        sig = np.array([svm_signal[t], xgb_signal[t]], dtype=np.float32)
        vec = np.concatenate([vec, sig])
    return vec


def get_state_dim(system_version: str) -> int:
    if system_version == "basic":
        return 1
    base = 18
    if system_version == "2.0":
        base += 2
    return base


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
