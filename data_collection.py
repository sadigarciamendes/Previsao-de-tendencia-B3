#Coleta de OHLCV da B3 via yfinance, com limpeza e seleção dos ativos


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
                    auto_adjust=cfg.auto_adjust, actions=cfg.actions,
                    rounding=cfg.rounding,
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


def load_dataset(cfg=CFG, use_cache: bool = True) -> Dict[str, pd.DataFrame]:

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    cache_path = os.path.join(cfg.outputs_dir, "prices_cache.pkl")

    if use_cache and os.path.exists(cache_path):
        data = pd.read_pickle(cache_path)
        print(f"[DATA] Cache carregado: {len(data)} ativos ({cache_path})")
    else:
        data = download_prices(cfg.tickers, cfg.data)
        pd.to_pickle(data, cache_path)
        print(f"[DATA] Cache salvo em {cache_path}")

    # O cache guarda tudo o que foi baixado; a exclusão é aplicada na leitura.
    removidos = [tk for tk in cfg.excluded if tk in data]
    data = {tk: df for tk, df in data.items() if tk not in cfg.excluded}
    for tk in removidos:
        print(f"[DATA] Excluído {tk}: {cfg.excluded[tk]}")
    print(f"[DATA] {len(data)} ativos após exclusões.")
    return data


if __name__ == "__main__":
    d = load_dataset(CFG)
    print(d[next(iter(d))].tail())
