from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dataset import AssetPanel


MIN_TREINO = 60      # janelas mínimas de treino interno para o fold valer
MIN_VAL = 20         # janelas mínimas de validação interna
MIN_TESTE = 10       # janelas mínimas de teste


@dataclass
class FoldRange:
    """Posições de treino, validação interna e teste de um ativo num fold."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class Fold:
    """Um fold: fronteiras de calendário e os recortes de cada ativo."""
    idx: int
    train_ate: pd.Timestamp
    teste_ate: pd.Timestamp
    por_ativo: Dict[str, FoldRange] = field(default_factory=dict)

    @property
    def n_treino(self) -> int:
        return sum(len(r.train) for r in self.por_ativo.values())

    @property
    def n_val(self) -> int:
        return sum(len(r.val) for r in self.por_ativo.values())

    @property
    def n_teste(self) -> int:
        return sum(len(r.test) for r in self.por_ativo.values())

    def periodo_teste(self, panels: Dict[str, AssetPanel]):
        """Primeira e última data efetivamente testadas no fold."""
        datas = [panels[tk].dates[r.test] for tk, r in self.por_ativo.items()
                 if len(r.test)]
        if not datas:
            return None, None
        return min(d[0] for d in datas), max(d[-1] for d in datas)


def calendar_boundaries(panels: Dict[str, AssetPanel], k: int) -> List[pd.Timestamp]:
    """k+1 fronteiras de data igualmente espaçadas na linha do tempo comum."""
    todas = np.unique(np.concatenate([p.dates.values for p in panels.values()]))
    n = len(todas)
    return [pd.Timestamp(todas[min(int(n * i / (k + 1)), n - 1)])
            for i in range(1, k + 2)]


def build_folds(panels: Dict[str, AssetPanel], k: int, gap: int,
                val_ratio: float) -> List[Fold]:
    """Monta os k folds walk-forward sobre o calendário comum aos ativos."""
    bounds = calendar_boundaries(panels, k)
    folds: List[Fold] = []

    for f in range(k):
        train_ate, teste_ate = bounds[f], bounds[f + 1]
        fold = Fold(idx=f + 1, train_ate=train_ate, teste_ate=teste_ate)

        for tk, p in panels.items():
            datas = p.dates.values
            # Região de treino: tudo antes da fronteira. Teste: até a próxima.
            a = int(np.searchsorted(datas, train_ate.to_datetime64(), side="left"))
            b = int(np.searchsorted(datas, teste_ate.to_datetime64(), side="right"))

            corte = int(a * (1.0 - val_ratio))
            train = np.arange(0, corte)
            val = np.arange(min(corte + gap, a), a)
            test = np.arange(min(a + gap, b), b)

            if (len(train) < MIN_TREINO or len(val) < MIN_VAL
                    or len(test) < MIN_TESTE):
                continue
            fold.por_ativo[tk] = FoldRange(train=train, val=val, test=test)

        if fold.por_ativo:
            folds.append(fold)

    if not folds:
        raise RuntimeError(
            "Nenhum fold válido. Reduza cv_folds ou verifique o histórico dos ativos.")
    return folds


def pool(panels: Dict[str, AssetPanel], fold: Fold, group: str,
         parte: str) -> Tuple[np.ndarray, np.ndarray]:
    """Junta, entre todos os ativos, as janelas de um grupo numa parte do fold."""
    Xs, ys = [], []
    for tk, r in fold.por_ativo.items():
        idx = getattr(r, parte)
        if len(idx) == 0:
            continue
        Xs.append(panels[tk].X_by_group[group][idx])
        ys.append(panels[tk].y[idx])
    if not Xs:
        raise RuntimeError(f"Fold {fold.idx}: parte '{parte}' vazia no grupo {group}.")
    return np.concatenate(Xs), np.concatenate(ys)

def describe_folds(folds: List[Fold], panels: Dict[str, AssetPanel]) -> List[dict]:
    """Resumo textual/numérico de cada fold, para o relatório."""
    linhas = []
    for fd in folds:
        ini, fim = fd.periodo_teste(panels)
        linhas.append({
            "fold": fd.idx,
            "n_ativos": len(fd.por_ativo),
            "treina_ate": str(fd.train_ate.date()),
            "teste_inicio": str(ini.date()) if ini is not None else None,
            "teste_fim": str(fim.date()) if fim is not None else None,
            "n_treino": fd.n_treino,
            "n_validacao": fd.n_val,
            "n_teste": fd.n_teste,
        })
    return linhas
