#Exporta os tensores prontos para a LSTM: um arquivo .npz por dobra em outputs/tensores/.

from __future__ import annotations
import os

import numpy as np

from config import CFG
from data_collection import load_dataset
from dataset import build_panels
from features import INDICATOR_GROUPS
from walkforward import build_folds, pool

PARTES = ("train", "val", "test")


# Ativo e data de cada janela, na mesma ordem em que pool() empilha os tensores.
def _meta(panels, fold, parte):
    tickers, datas = [], []
    for tk, r in fold.por_ativo.items():
        idx = getattr(r, parte)
        if len(idx) == 0:
            continue
        tickers.append(np.full(len(idx), tk.replace(".SA", ""), dtype="<U12"))
        datas.append(panels[tk].dates.values[idx].astype("datetime64[D]"))
    return np.concatenate(tickers), np.concatenate(datas)


def exportar(cfg=CFG):
    pasta = os.path.join(cfg.outputs_dir, "tensores")
    os.makedirs(pasta, exist_ok=True)

    # Mesmo pipeline do process_data.py: preços -> janelas -> dobras.
    data = load_dataset(cfg)
    panels, feat_names = build_panels(data, cfg)
    folds = build_folds(panels, cfg.split.cv_folds, cfg.data.forecast_horizon,
                        cfg.split.val_ratio)

    for fd in folds:
        arrays = {}
        for parte in PARTES:
            # Um X por grupo (entrada de cada especialista); o y é o mesmo para todos.
            for g in INDICATOR_GROUPS:
                X, y = pool(panels, fd, g, parte)
                arrays[f"X_{g}_{parte}"] = X.astype(np.float32)
            arrays[f"y_{parte}"] = y.astype(np.int8)  # -1 baixa, 0 neutra, +1 alta
            arrays[f"ticker_{parte}"], arrays[f"data_{parte}"] = _meta(panels, fd, parte)
        # Nomes das features (ordem das colunas de X) e fronteiras da dobra.
        for g in INDICATOR_GROUPS:
            arrays[f"features_{g}"] = np.array(feat_names[g])
        arrays["treina_ate"] = np.datetime64(fd.train_ate.date(), "D")
        arrays["teste_ate"] = np.datetime64(fd.teste_ate.date(), "D")

        # Comprimido: janelas vizinhas repetem 29 de 30 linhas, então o arquivo fica pequeno.
        caminho = os.path.join(pasta, f"fold{fd.idx}.npz")
        np.savez_compressed(caminho, **arrays)
        mb = os.path.getsize(caminho) / 2**20
        print(f"fold {fd.idx}: treino={len(arrays['y_train']):,} "
              f"val={len(arrays['y_val']):,} teste={len(arrays['y_test']):,} "
              f"-> {caminho} ({mb:.0f} MB)")


if __name__ == "__main__":
    exportar(CFG)
