#Executa o pipeline de dados e salva um resumo em outputs/.


from __future__ import annotations
import json
import os

import numpy as np

from config import CFG
from data_collection import load_dataset
from features import INDICATOR_GROUPS
from dataset import build_panels
from walkforward import build_folds, describe_folds


def main(cfg=CFG):
    print("=== Pipeline de dados B3 ===")

    data = load_dataset(cfg)
    panels, feat_names = build_panels(data, cfg)

    print("\nTensores por grupo (série completa, sem corte de avaliação):")
    resumo = {"ativos_validos": len(panels),
              "ativos_excluidos": dict(cfg.excluded), "grupos": {}}
    for g in INDICATOR_GROUPS:
        X = np.concatenate([p.X_by_group[g] for p in panels.values()])
        fmin, fmax = float(X.min()), float(X.max())
        ok = (fmin >= -1.0001) and (fmax <= 1.0001)
        print(f"  {g:7s} {X.shape} range=[{fmin:.3f},{fmax:.3f}] "
              f"{'OK' if ok else 'FORA DE [-1,1]!'}")
        resumo["grupos"][g] = {
            "shape": list(X.shape),
            "feature_names": feat_names[g],
            "range": [round(fmin, 4), round(fmax, 4)],
        }

    y_all = np.concatenate([p.y for p in panels.values()])
    nomes = {-1: "baixa", 0: "neutra", 1: "alta"}
    print("\nDistribuição das classes (-1 baixa, 0 neutra, +1 alta):")
    dist = {}
    for c in (-1, 0, 1):
        frac = float((y_all == c).mean()) if y_all.size else float("nan")
        print(f"  {nomes[c]:7s} = {frac:.1%}")
        dist[nomes[c]] = round(frac, 4)
    resumo["distribuicao_classes"] = dist

    folds = build_folds(panels, cfg.split.cv_folds, cfg.data.forecast_horizon,
                        cfg.split.val_ratio)
    descricao = describe_folds(folds, panels)
    print(f"\nValidação cruzada com avanço temporal — {len(folds)} folds:")
    for d in descricao:
        print(f"  fold {d['fold']}: treina até {d['treina_ate']} | "
              f"testa {d['teste_inicio']} a {d['teste_fim']} | "
              f"treino={d['n_treino']:,} val={d['n_validacao']:,} "
              f"teste={d['n_teste']:,} | {d['n_ativos']} ativos")
    resumo["folds"] = descricao

    p = panels.get(cfg.focus_ticker)
    if p is not None:
        print(f"\nAtivo de foco {cfg.focus_ticker}: {len(p)} janelas "
              f"({p.dates[0].date()} a {p.dates[-1].date()})")
        resumo["ativo_foco"] = {
            "ticker": cfg.focus_ticker, "janelas": len(p),
            "inicio": str(p.dates[0].date()), "fim": str(p.dates[-1].date()),
        }

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    out_path = os.path.join(cfg.outputs_dir, "resumo_dados.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2)
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main(CFG)
