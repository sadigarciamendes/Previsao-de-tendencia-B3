#Exporta o dataframe processado de um ativo para CSV.

#Uso:
   # python export_csv.py           
   # python export_csv.py VALE3.SA


from __future__ import annotations
import argparse
import os

import pandas as pd

from config import CFG
from data_collection import download_prices
from dataset import _align_stock
from features import INDICATOR_GROUPS


def export_ticker_csv(ticker: str, cfg=CFG, out_dir: str | None = None) -> str:
    """Baixa, processa e salva em CSV o dataframe de um ativo.

    Colunas: close, os atributos de cada grupo e label (-1 baixa, 0 neutra, +1 alta).
    """
    out_dir = out_dir or cfg.outputs_dir

    raw = download_prices([ticker], cfg.data)
    if ticker not in raw:
        raise SystemExit(f"Sem dados válidos para {ticker}.")

    groups, target, close, _dates = _align_stock(raw[ticker], cfg.data)

    df = pd.concat([groups[g] for g in INDICATOR_GROUPS], axis=1).round(6)
    df.insert(0, "close", close.round(2))
    df["label"] = target.astype(int)
    df.index.name = "data"

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dados_{ticker.replace('.SA', '')}.csv")
    df.to_csv(path, encoding="utf-8")

    print(f"[CSV] {path} | {df.shape[0]} linhas x {df.shape[1]} colunas")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exporta o dataframe processado de um ativo da B3 para CSV.")
    parser.add_argument("ticker", nargs="?", default=CFG.focus_ticker)
    args = parser.parse_args()
    export_ticker_csv(args.ticker, CFG)
