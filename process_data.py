# -*- coding: utf-8 -*-
"""Executa o pipeline de dados e salva um resumo em outputs/.

Coleta -> indicadores em força [-1,1] -> rotulagem em 3 classes -> janelas ->
tensores. Imprime as dimensões, valida o intervalo [-1,1] e a distribuição das
classes.
"""

from __future__ import annotations
import json
import os

import numpy as np

from config import CFG
from data_collection import load_dataset
from features import talib_status
from dataset import build_supervised


def main(cfg=CFG):
    print(f"=== Pipeline de dados B3 | {talib_status()} ===")

    data = load_dataset(cfg)
    tensors, cv, holdout, slices = build_supervised(data, cfg)

    print("\nTensores por grupo:")
    resumo = {"ativos_validos": len(data), "grupos": {}}
    for g, t in tensors.items():
        fmin = float(t.X_train.min()) if t.X_train.size else float("nan")
        fmax = float(t.X_train.max()) if t.X_train.size else float("nan")
        ok = (fmin >= -1.0001) and (fmax <= 1.0001)
        print(f"  {g:7s} treino={t.X_train.shape} holdout={t.X_test.shape} "
              f"range=[{fmin:.3f},{fmax:.3f}] {'OK' if ok else 'FORA DE [-1,1]!'}")
        resumo["grupos"][g] = {
            "treino": list(t.X_train.shape),
            "holdout": list(t.X_test.shape),
            "feature_names": t.feature_names,
            "range": [round(fmin, 4), round(fmax, 4)],
        }

    g0 = next(iter(tensors.values()))
    y_all = np.concatenate([g0.y_train, g0.y_test])
    nomes = {-1: "baixa", 0: "neutra", 1: "alta"}
    print("\nDistribuição das classes (-1 baixa, 0 neutra, +1 alta):")
    dist = {}
    for c in (-1, 0, 1):
        frac = float((y_all == c).mean()) if y_all.size else float("nan")
        print(f"  {nomes[c]:7s} = {frac:.1%}")
        dist[nomes[c]] = round(frac, 4)
    resumo["distribuicao_classes"] = dist

    if holdout is not None:
        ini, fim = holdout.dates[0].date(), holdout.dates[-1].date()
        print(f"\nHold-out {cfg.focus_ticker}: {len(holdout.dates)} amostras ({ini} a {fim})")
        resumo["holdout_focus"] = {
            "ticker": cfg.focus_ticker,
            "amostras": len(holdout.dates),
            "inicio": str(ini), "fim": str(fim),
        }

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    out_path = os.path.join(cfg.outputs_dir, "resumo_dados.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2)
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main(CFG)
