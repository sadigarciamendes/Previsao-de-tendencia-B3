#Indicadores técnicos como índices de força limitados a [-1, 1].

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

INDICATOR_GROUPS: List[str] = ["RSI", "MACD", "MA", "BBANDS", "OBV"]

_EPS: float = 1e-12


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _bbands(close: pd.Series, period=20, ndev=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return mid + ndev * std, mid, mid - ndev * std


def build_rsi_features(close: pd.Series) -> pd.DataFrame:
    rsi = _rsi(close, 14)

    f = pd.DataFrame(index=close.index)
    f["rsi_force"] = (rsi - 50.0) / 50.0
    f["rsi_oversold"] = (rsi < 30).astype(float)
    f["rsi_overbought"] = (rsi > 70).astype(float)
    f["rsi_above50"] = (rsi > 50).astype(float)
    f["rsi_cross50"] = np.sign(rsi - 50).diff().fillna(0.0).clip(-1, 1)
    return f


def build_macd_features(close: pd.Series) -> pd.DataFrame:
    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)

    macd_force = (ema_fast - ema_slow) / (ema_fast + ema_slow + _EPS)
    signal_force = _ema(macd_force, 9)
    hist = macd_force - signal_force

    f = pd.DataFrame(index=close.index)
    f["macd_force"] = macd_force.clip(-1, 1)
    f["macd_signal_force"] = signal_force.clip(-1, 1)
    f["macd_hist"] = (hist / 2.0).clip(-1, 1)
    f["macd_hist_dir"] = np.sign(hist.diff().fillna(0.0)).clip(-1, 1)
    f["macd_cross"] = np.sign(hist).diff().fillna(0.0).clip(-1, 1)
    return f


def build_ma_features(close: pd.Series) -> pd.DataFrame:
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema9 = _ema(close, 9)
    ema20 = _ema(close, 20)

    f = pd.DataFrame(index=close.index)
    f["price_sma50_force"] = (close - sma50) / (close + sma50 + _EPS)
    f["price_sma200_force"] = (close - sma200) / (close + sma200 + _EPS)
    f["sma_trend_force"] = (sma50 - sma200) / (sma50 + sma200 + _EPS)
    f["ema_short_force"] = (ema9 - ema20) / (ema9 + ema20 + _EPS)
    f["golden_death_cross"] = np.sign(sma50 - sma200).diff().fillna(0.0).clip(-1, 1)
    f["ema_cross"] = np.sign(ema9 - ema20).diff().fillna(0.0).clip(-1, 1)
    return f


def build_bbands_features(close: pd.Series) -> pd.DataFrame:
    upper, mid, lower = _bbands(close, 20, 2.0)

    f = pd.DataFrame(index=close.index)
    f["bb_position"] = ((close - mid) / ((upper - mid) + _EPS)).clip(-1, 1)
    bw = (upper - lower) / (mid + _EPS)
    bw_ref = bw.rolling(50, min_periods=10).mean()
    f["bb_vol_force"] = 2.0 * bw / (bw + bw_ref + _EPS) - 1.0
    f["bb_above_upper"] = (close > upper).astype(float)
    f["bb_below_lower"] = (close < lower).astype(float)
    return f


def build_obv_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()

    obv_ema_s = _ema(obv, 10)
    obv_ema_l = _ema(obv, 30)
    denom = obv_ema_s.abs() + obv_ema_l.abs() + _EPS

    f = pd.DataFrame(index=close.index)
    f["obv_trend_force"] = ((obv_ema_s - obv_ema_l) / denom).clip(-1, 1)
    f["obv_cross"] = np.sign(obv_ema_s - obv_ema_l).diff().fillna(0.0).clip(-1, 1)
    vol_ma = volume.rolling(20, min_periods=5).mean()
    f["vol_force"] = ((volume - vol_ma) / (volume + vol_ma + _EPS)).clip(-1, 1)
    f["vol_high"] = (volume > 2.0 * vol_ma).astype(float)
    return f


def build_all_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calcula os cinco grupos de indicadores para um ativo."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    return {
        "RSI": build_rsi_features(close),
        "MACD": build_macd_features(close),
        "MA": build_ma_features(close),
        "BBANDS": build_bbands_features(close),
        "OBV": build_obv_features(close, volume),
    }
