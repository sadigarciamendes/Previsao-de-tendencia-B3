from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CFG, DataConfig
from features import build_all_features, INDICATOR_GROUPS


@dataclass
class GroupTensors:
    """Tensores de um grupo: treino (pool) e hold-out agregado."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


@dataclass
class GroupCV:
    """Janelas de treino de um grupo, separadas por ativo e em ordem temporal,
    para a validação cruzada com avanço temporal."""
    per_asset: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


@dataclass
class HoldoutSlice:
    """Recorte de hold-out de um ativo: datas, preços, janelas e rótulos."""
    dates: pd.DatetimeIndex
    close: np.ndarray
    X_by_group: Dict[str, np.ndarray]
    y_true: np.ndarray


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


def build_supervised(
    data: Dict[str, pd.DataFrame],
    cfg=CFG,
) -> Tuple[Dict[str, GroupTensors], Dict[str, GroupCV],
           Optional[HoldoutSlice], Dict[str, HoldoutSlice]]:
    """Gera os tensores por grupo, as janelas de treino por ativo e os hold-outs.

    Retorna (tensors_by_group, cv_by_group, holdout_do_focus, slices_por_ativo).
    """
    window = cfg.data.window_size
    groups = INDICATOR_GROUPS

    aligned: Dict[str, dict] = {}
    for tk, df in data.items():
        g_al, y_al, close_al, dates = _align_stock(df, cfg.data)
        n = len(dates)
        if n <= window + 5:
            continue
        split = int(n * (1.0 - cfg.data.holdout_ratio))
        aligned[tk] = dict(groups=g_al, target=y_al.values,
                           close=close_al.values, dates=dates, split=split)

    if not aligned:
        raise RuntimeError("Nenhum ativo com dados suficientes após alinhamento.")

    feat_names: Dict[str, List[str]] = {
        g: list(next(iter(aligned.values()))["groups"][g].columns) for g in groups
    }

    pooled: Dict[str, dict] = {g: dict(Xtr=[], ytr=[], Xte=[], yte=[]) for g in groups}
    cv: Dict[str, GroupCV] = {g: GroupCV(feature_names=feat_names[g]) for g in groups}
    holdout: Optional[HoldoutSlice] = None
    slices: Dict[str, HoldoutSlice] = {}
    focus_ticker = cfg.focus_ticker

    for tk, info in aligned.items():
        split = info["split"]
        n = len(info["dates"])
        target = info["target"]
        tk_X_by_group: Dict[str, np.ndarray] = {}
        tk_pos_ref = None

        for g in groups:
            feat = np.nan_to_num(info["groups"][g].values, nan=0.0,
                                 posinf=0.0, neginf=0.0).astype(np.float32)
            Xtr, ytr, _ = _make_windows(feat, target, window, 0, split)
            Xho, yho, pos_ho = _make_windows(feat, target, window, split, n)

            pooled[g]["Xtr"].append(Xtr); pooled[g]["ytr"].append(ytr)
            pooled[g]["Xte"].append(Xho); pooled[g]["yte"].append(yho)
            if len(Xtr) > 0:
                cv[g].per_asset.append((Xtr, ytr))

            tk_X_by_group[g] = Xho
            tk_pos_ref = pos_ho

        if tk_pos_ref is not None and len(tk_pos_ref) > 0:
            sl = HoldoutSlice(
                dates=info["dates"][tk_pos_ref],
                close=info["close"][tk_pos_ref],
                X_by_group=tk_X_by_group,
                y_true=target[tk_pos_ref].astype(np.float32),
            )
            slices[tk] = sl
            if tk == focus_ticker:
                holdout = sl

    out: Dict[str, GroupTensors] = {}
    for g in groups:
        out[g] = GroupTensors(
            X_train=np.concatenate(pooled[g]["Xtr"]),
            y_train=np.concatenate(pooled[g]["ytr"]),
            X_test=np.concatenate(pooled[g]["Xte"]),
            y_test=np.concatenate(pooled[g]["yte"]),
            feature_names=feat_names[g],
        )

    return out, cv, holdout, slices


if __name__ == "__main__":
    from data_collection import load_dataset
    d = load_dataset(CFG)
    tensors, cv, holdout, slices = build_supervised(d, CFG)
    for g, t in tensors.items():
        print(f"{g:7s} | treino={t.X_train.shape} holdout={t.X_test.shape} "
              f"| ativos={len(cv[g].per_asset)}")
    if holdout:
        print(f"Hold-out {CFG.focus_ticker}: {len(holdout.dates)} amostras")
