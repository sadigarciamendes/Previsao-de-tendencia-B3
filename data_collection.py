# Coleta de OHLCV da B3 via yfinance, com limpeza e seleção dos ativos

from __future__ import annotations
import os
import time
import warnings
from typing import Dict, List

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Instale o yfinance: pip install yfinance") from e

from config import CFG, DataConfig


OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download_prices(tickers: List[str], cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    """Baixa os candles diários de cada ticker e retorna {ticker: DataFrame OHLCV}.

    Ativos sem dados ou com histórico menor que `min_history` são descartados.
    """
    period = f"{cfg.period_years}y"
    raw: Dict[str, pd.DataFrame] = {}
    max_retries = 4
    base_delay = 1.5

    print(f"[DATA] Baixando {len(tickers)} ativos | período={period} | "
          f"intervalo={cfg.interval}")

    for tk in tickers:
        df = None
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(
                    tk, period=period, interval=cfg.interval,
                    auto_adjust=False, actions=True, rounding=True,
                    progress=False, threads=False,
                )
            except Exception as exc:
                df = None
                if attempt == max_retries:
                    warnings.warn(f"Falha ao baixar {tk}: {exc}")
            if df is not None and not df.empty:
                break
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))

        if df is None or df.empty:
            warnings.warn(f"{tk}: sem dados após {max_retries} tentativas.")
            time.sleep(base_delay)
            continue

        time.sleep(base_delay)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[[c for c in OHLCV if c in df.columns]].copy().dropna()
        if len(df) < cfg.min_history:
            warnings.warn(f"{tk}: histórico curto ({len(df)} candles). Ignorado.")
            continue

        raw[tk] = df

    print(f"[DATA] {len(raw)} ativos válidos após limpeza.")
    return raw


def load_dataset(cfg=CFG) -> Dict[str, pd.DataFrame]:
    """Baixa e limpa todos os tickers configurados."""
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    return download_prices(cfg.tickers, cfg.data)


if __name__ == "__main__":
    d = load_dataset(CFG)
    print(d[next(iter(d))].tail())
