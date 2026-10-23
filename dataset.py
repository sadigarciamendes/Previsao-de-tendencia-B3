from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CFG, DataConfig
from features import build_all_features, INDICATOR_GROUPS


@dataclass
class AssetPanel:
    """Série completa de janelas de um ativo, sem nenhum corte de avaliação.

    Cada posição `i` corresponde a uma janela cujo último pregão é `dates[i]`,
    com preço `close[i]` e rótulo `y[i]`. Os cortes de treino/validação/teste são
    feitos depois, por fold, em `walkforward.py`.
    """
    ticker: str
    dates: pd.DatetimeIndex
    close: np.ndarray
    y: np.ndarray
    X_by_group: Dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "AssetPanel":
        """Recorta o painel nas posições `idx` (mantém a ordem temporal)."""
        return AssetPanel(
            ticker=self.ticker,
            dates=self.dates[idx],
            close=self.close[idx],
            y=self.y[idx],
            X_by_group={g: X[idx] for g, X in self.X_by_group.items()},
        )


def make_labels(close: pd.Series, horizon: int, neutral_k: float) -> pd.Series:
    """Rótulo de tendência em três classes (-1 baixa, 0 neutra, +1 alta).

    O retorno futuro de `horizon` pregões é comparado a um limiar que acompanha a
    volatilidade local: limiar = neutral_k * desvio dos retornos diários * sqrt(horizon).
    Movimentos dentro da banda recebem a classe neutra.
    """
    fwd_ret = close.shift(-horizon) / close - 1.0
    daily_vol = close.pct_change().rolling(20, min_periods=10).std()
    threshold = neutral_k * daily_vol * np.sqrt(horizon)

    label = pd.Series(0.0, index=close.index)
    label[fwd_ret > threshold] = 1.0
    label[fwd_ret < -threshold] = -1.0
    label[fwd_ret.isna() | threshold.isna()] = np.nan
    return label


def _align_stock(df: pd.DataFrame, cfg: DataConfig):
    """Alinha features e rótulo no mesmo índice, removendo warm-up e cauda sem alvo."""
    groups = build_all_features(df)
    target = make_labels(df["Close"].astype(float), cfg.forecast_horizon, cfg.neutral_k)

    common = df.index
    for g in groups.values():
        common = common.intersection(g.dropna().index)
    common = common.intersection(target.dropna().index).sort_values()

    groups_aligned = {name: g.loc[common] for name, g in groups.items()}
    return groups_aligned, target.loc[common], df["Close"].astype(float).loc[common], common


def _make_windows(feat: np.ndarray, target: np.ndarray, window: int,
                  start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria janelas cujo último índice t está em [start, end), com y = target[t]."""
    xs, ys, pos = [], [], []
    for t in range(max(start, window - 1), end):
        xs.append(feat[t - window + 1: t + 1])
        ys.append(target[t])
        pos.append(t)
    if not xs:
        empty = np.empty((0, window, feat.shape[1]), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(pos, dtype=int))


def build_panels(
    data: Dict[str, pd.DataFrame],
    cfg=CFG,
) -> Tuple[Dict[str, AssetPanel], Dict[str, List[str]]]:
    """Monta o painel completo de cada ativo e os nomes de feature de cada grupo.

    Não há corte de treino/teste aqui: a série de janelas cobre todo o período
    alinhado. A separação é feita por fold, por data de calendário, em
    `walkforward.py`.
    """
    window = cfg.data.window_size
    groups = INDICATOR_GROUPS

    panels: Dict[str, AssetPanel] = {}
    feat_names: Dict[str, List[str]] = {}

    for tk, df in data.items():
        g_al, y_al, close_al, dates = _align_stock(df, cfg.data)
        if len(dates) <= window + 5:
            continue

        target = y_al.values
        X_by_group: Dict[str, np.ndarray] = {}
        pos_ref = None
        for g in groups:
            if g not in feat_names:
                feat_names[g] = list(g_al[g].columns)
            feat = np.nan_to_num(g_al[g].values, nan=0.0,
                                 posinf=0.0, neginf=0.0).astype(np.float32)
            X, _, pos = _make_windows(feat, target, window, 0, len(dates))
            X_by_group[g] = X
            pos_ref = pos

        if pos_ref is None or len(pos_ref) == 0:
            continue

        panels[tk] = AssetPanel(
            ticker=tk,
            dates=dates[pos_ref],
            close=close_al.values[pos_ref],
            y=target[pos_ref].astype(np.float32),
            X_by_group=X_by_group,
        )

    if not panels:
        raise RuntimeError("Nenhum ativo com dados suficientes após alinhamento.")
    return panels, feat_names


if __name__ == "__main__":
    from data_collection import load_dataset
    d = load_dataset(CFG)
    panels, feat_names = build_panels(d, CFG)
    print(f"{len(panels)} ativos com painel montado.")
    for g, names in feat_names.items():
        print(f"  {g:7s} {len(names)} features: {', '.join(names)}")
    p = panels.get(CFG.focus_ticker) or next(iter(panels.values()))
    print(f"\n{p.ticker}: {len(p)} janelas de "
          f"{p.dates[0].date()} a {p.dates[-1].date()}")
    for g, X in p.X_by_group.items():
        print(f"  {g:7s} {X.shape}")
