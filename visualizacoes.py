#Figuras da etapa de dados (para o artigo). Salva PNGs em outputs/figuras/.
#
#Uso:  python visualizacoes.py            (ativo de foco do config)
#      python visualizacoes.py VALE3.SA   (outro ativo nas figuras por ativo)

from __future__ import annotations
import locale
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

from config import CFG
from data_collection import load_dataset
from dataset import build_panels
from features import INDICATOR_GROUPS, build_all_features
from walkforward import build_folds


# Paleta (categórica em ordem fixa; classes = par divergente vermelho/cinza/azul).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CLASSE_COR = {-1: "#e34948", 0: "#a3a29b", 1: "#2a78d6"}
CLASSE_NOME = {-1: "Baixa (−1)", 0: "Neutra (0)", 1: "Alta (+1)"}
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"

CONTINUAS = ["rsi_force", "macd_force", "macd_signal_force", "macd_hist",
             "price_sma50_force", "price_sma200_force", "sma_trend_force",
             "ema_short_force", "bb_position", "bb_vol_force",
             "obv_trend_force", "vol_force"]


def _estilo():
    for loc in ("pt_BR.UTF-8", "Portuguese_Brazil.1252", "pt_BR"):
        try:
            locale.setlocale(locale.LC_NUMERIC, loc)
            plt.rcParams["axes.formatter.use_locale"] = True
            break
        except locale.Error:
            continue
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": TINTA_2,
        "axes.edgecolor": TINTA_2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "legend.frameon": False, "lines.linewidth": 1.4, "axes.axisbelow": True,
    })


def _num(v, casas=None) -> str:
    """Número com vírgula decimal."""
    txt = f"{v:.{casas}f}" if casas is not None else f"{v}"
    return txt.replace(".", ",")


def _datas(ax, fmt="%m/%Y"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _salvar(fig, nome, pasta):
    caminho = os.path.join(pasta, nome)
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  {caminho}")


# ---------------------------------------------------------------------------
# Figuras por ativo
# ---------------------------------------------------------------------------

def fig_precos_medias(df, tk, pasta):
    """Close bruto com SMA50/SMA200 e os cruzamentos golden/death cross."""
    close = df["Close"].astype(float)
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    cruz = np.sign(sma50 - sma200).diff()
    golden, death = cruz[cruz > 0].index, cruz[cruz < 0].index

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(close.index, close, color=SERIES[0], lw=1.0, label="Fechamento (bruto)")
    ax.plot(sma50.index, sma50, color=SERIES[1], label="SMA50")
    ax.plot(sma200.index, sma200, color=SERIES[2], label="SMA200")
    ax.scatter(golden, sma50.loc[golden], marker="^", s=60, color=TINTA,
               zorder=5, label="Golden cross (+1)")
    ax.scatter(death, sma50.loc[death], marker="v", s=60, facecolor="white",
               edgecolor=TINTA, linewidth=1.2, zorder=5, label="Death cross (−1)")
    ax.set_title(f"{tk.replace('.SA', '')} — preço e médias móveis longas")
    ax.set_ylabel("R$")
    _datas(ax, "%Y")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.1))
    _salvar(fig, "fig01_preco_medias.png", pasta)


def fig_indicadores(df, tk, pasta, n=250):
    """Principais índices de força de cada grupo, último ~1 ano."""
    g = {k: v.iloc[-n:] for k, v in build_all_features(df).items()}
    paineis = [
        ("RSI", [("rsi_force", "rsi_force")]),
        ("MACD", [("macd_force", "macd_force"),
                  ("macd_signal_force", "macd_signal_force")]),
        ("MA", [("sma_trend_force", "sma_trend_force (SMA50×SMA200)"),
                ("ema_short_force", "ema_short_force (EMA9×EMA20)")]),
        ("BBANDS", [("bb_position", "bb_position"),
                    ("bb_vol_force", "bb_vol_force")]),
        ("OBV", [("obv_trend_force", "obv_trend_force"),
                 ("vol_force", "vol_force")]),
    ]
    fig, axs = plt.subplots(len(paineis), 1, figsize=(11, 9), sharex=True)
    for ax, (grupo, cols) in zip(axs, paineis):
        for i, (c, rot) in enumerate(cols):
            s = g[grupo][c]
            ax.plot(s.index, s, color=SERIES[i], label=rot)
        ax.axhline(0, color=TINTA_2, lw=0.6)
        ax.set_title(grupo)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _datas(axs[-1])
    fig.suptitle(f"{tk.replace('.SA', '')} — índices de força por grupo "
                 f"(últimos {n} pregões)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _salvar(fig, "fig02_indicadores.png", pasta)


def fig_rotulo(df, tk, pasta, n=250):
    """Retorno de h pregões à frente contra a banda ±τ que define as classes."""
    h, k = CFG.data.forecast_horizon, CFG.data.neutral_k
    close = df["Close"].astype(float)
    fwd = close.shift(-h) / close - 1.0
    tau = k * close.pct_change().rolling(20, min_periods=10).std() * np.sqrt(h)
    d = pd.DataFrame({"fwd": fwd, "tau": tau}).dropna().iloc[-n:]
    classe = np.where(d.fwd > d.tau, 1, np.where(d.fwd < -d.tau, -1, 0))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(d.index, -d.tau * 100, d.tau * 100, color="#f0efec",
                    label="Banda neutra ±τ")
    ax.plot(d.index, d.tau * 100, color=TINTA_2, lw=0.8)
    ax.plot(d.index, -d.tau * 100, color=TINTA_2, lw=0.8)
    for c in (-1, 0, 1):
        m = classe == c
        ax.scatter(d.index[m], d.fwd[m] * 100, s=10, color=CLASSE_COR[c],
                   label=CLASSE_NOME[c], zorder=3)
    ax.axhline(0, color=TINTA_2, lw=0.6)
    ax.set_title(f"{tk.replace('.SA', '')} — rotulagem: retorno de {h} pregões "
                 f"vs. τ = {_num(k)}·σ₂₀·√{h}")
    ax.set_ylabel("Retorno futuro (%)")
    _datas(ax)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    _salvar(fig, "fig03_rotulagem.png", pasta)


def fig_janela(panel, feat_names, pasta):
    """Mapa de calor de uma janela de entrada (30 pregões × 24 features)."""
    X = np.concatenate([panel.X_by_group[g][-1] for g in INDICATOR_GROUPS], axis=1)
    nomes = [n for g in INDICATOR_GROUPS for n in feat_names[g]]
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#e34948", "#f0efec", "#2a78d6"])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(X.T, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(len(nomes)), nomes, fontsize=8)
    w = X.shape[0]
    ax.set_xticks(range(0, w, 5), [f"t−{w - 1 - i}" if i < w - 1 else "t"
                                   for i in range(0, w, 5)])
    pos = 0
    for g in INDICATOR_GROUPS[:-1]:
        pos += len(feat_names[g])
        ax.axhline(pos - 0.5, color=TINTA, lw=1.2)
    ax.set_xlabel("Posição na janela (pregões)")
    fim = panel.dates[-1].strftime("%d/%m/%Y")
    ax.set_title(f"{panel.ticker.replace('.SA', '')} — janela de entrada "
                 f"terminada em {fim} (rótulo = {int(panel.y[-1]):+d})")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    _salvar(fig, "fig08_janela_entrada.png", pasta)


# ---------------------------------------------------------------------------
# Figuras do conjunto completo
# ---------------------------------------------------------------------------

def fig_classes(panels, folds, pasta):
    """Proporção das classes no total e no bloco de teste de cada dobra."""
    linhas = [("Total", np.concatenate([p.y for p in panels.values()]))]
    for fd in folds:
        ys = [panels[tk].y[r.test] for tk, r in fd.por_ativo.items()]
        linhas.append((f"Dobra {fd.idx} (teste)", np.concatenate(ys)))

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (_, y) in enumerate(linhas[::-1]):
        esq = 0.0
        for c in (-1, 0, 1):
            f = float((y == c).mean()) * 100
            ax.barh(i, f, left=esq, color=CLASSE_COR[c], height=0.7,
                    edgecolor="white", linewidth=2,
                    label=CLASSE_NOME[c] if i == 0 else None)
            ax.text(esq + f / 2, i, f"{f:.1f}%".replace(".", ","), ha="center",
                    va="center", fontsize=8,
                    color="white" if c != 0 else TINTA)
            esq += f
    ax.set_yticks(range(len(linhas)), [r for r, _ in linhas[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% das janelas")
    ax.grid(False)
    ax.set_title("Distribuição das classes")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18))
    _salvar(fig, "fig04_classes.png", pasta)


def fig_walkforward(panels, folds, pasta):
    """Linha do tempo das dobras: treino, validação, vão e teste."""
    ini = min(p.dates[0] for p in panels.values())
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, fd in enumerate(folds):
        val_ini = pd.Series([panels[tk].dates[r.val[0]]
                             for tk, r in fd.por_ativo.items()]).median()
        t0, t1 = fd.periodo_teste(panels)
        y = len(folds) - 1 - i
        partes = [(ini, val_ini, SERIES[0], "Treino"),
                  (val_ini, fd.train_ate, SERIES[1], "Validação (15% final)"),
                  (t0, t1, SERIES[2], "Teste")]
        for a, b, cor, rot in partes:
            ax.barh(y, mdates.date2num(b) - mdates.date2num(a),
                    left=mdates.date2num(a), color=cor, height=0.6,
                    label=rot if i == 0 else None)
        ax.text(mdates.date2num(t1) + 20, y,
                f"{fd.n_teste:,} jan. | {len(fd.por_ativo)} ativos".replace(",", "."),
                va="center", fontsize=8, color=TINTA_2)
    ax.set_yticks(range(len(folds)), [f"Dobra {fd.idx}" for fd in folds][::-1])
    ax.xaxis_date()
    _datas(ax, "%Y")
    ax.set_xlim(right=mdates.date2num(max(p.dates[-1] for p in panels.values()))
                + 520)
    ax.grid(axis="y", visible=False)
    ax.set_title("Validação cruzada walk-forward (vão de "
                 f"{CFG.data.forecast_horizon} janelas entre as partes)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    _salvar(fig, "fig05_walkforward.png", pasta)


def fig_ativos_tempo(panels, folds, pasta):
    """Número de ativos com janela disponível em cada data."""
    cont = pd.Series(np.concatenate([p.dates.values for p in panels.values()]))
    cont = cont.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(cont.index, cont.values, color=SERIES[0], drawstyle="steps-post")
    for fd in folds:
        ax.axvline(fd.train_ate, color=TINTA_2, lw=0.8, ls="--")
        ax.text(fd.train_ate, 2, f" D{fd.idx}", fontsize=8, color=TINTA_2)
    ax.set_ylim(0, len(panels) + 5)
    ax.set_ylabel("Ativos")
    ax.set_title("Ativos com janela disponível ao longo do tempo "
                 "(tracejado = início do teste de cada dobra)")
    _datas(ax, "%Y")
    _salvar(fig, "fig06_ativos_no_tempo.png", pasta)


def fig_limiar(data, panels, pasta):
    """Histograma do limiar τ em todas as janelas."""
    h, k = CFG.data.forecast_horizon, CFG.data.neutral_k
    taus = []
    for tk, p in panels.items():
        close = data[tk]["Close"].astype(float)
        tau = k * close.pct_change().rolling(20, min_periods=10).std() * np.sqrt(h)
        taus.append(tau.loc[p.dates].values)
    tau = np.concatenate(taus) * 100
    media, mediana = tau.mean(), np.median(tau)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(tau, bins=np.linspace(0, np.percentile(tau, 99.5), 80),
            color=SERIES[0], edgecolor="white", linewidth=0.4)
    topo = ax.get_ylim()[1] * 0.97
    for v, rot, ls, ha in ((media, "média", "-", "left"),
                           (mediana, "mediana", "--", "right")):
        ax.axvline(v, color=TINTA, lw=1, ls=ls)
        ax.text(v, topo, f" {rot} {_num(v, 2)}% ", fontsize=8, va="top", ha=ha)
    ax.set_xlabel("τ (%)")
    ax.set_ylabel("Janelas")
    n_jan = f"{len(tau):,}".replace(",", ".")
    ax.set_title(f"Distribuição do limiar τ = {_num(k)}·σ₂₀·√{h} "
                 f"({n_jan} janelas)")
    _salvar(fig, "fig07_limiar_tau.png", pasta)


def fig_features(panels, feat_names, pasta, amostra=30000, seed=0):
    """Escala das features: caixas das contínuas e frequência das discretas."""
    nomes = [n for g in INDICATOR_GROUPS for n in feat_names[g]]
    grupo_de = {n: g for g in INDICATOR_GROUPS for n in feat_names[g]}
    X = np.concatenate([np.concatenate([p.X_by_group[g][:, -1, :]
                                        for g in INDICATOR_GROUPS], axis=1)
                        for p in panels.values()])
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), size=min(amostra, len(X)), replace=False)]
    df = pd.DataFrame(X, columns=nomes)
    cor_g = {g: SERIES[i] for i, g in enumerate(INDICATOR_GROUPS)}
    disc = [n for n in nomes if n not in CONTINUAS]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    cont = CONTINUAS[::-1]
    bp = a1.boxplot([df[c] for c in cont], vert=False, whis=(1, 99),
                    showfliers=False, patch_artist=True, widths=0.6,
                    medianprops={"color": TINTA, "lw": 1})
    for patch, c in zip(bp["boxes"], cont):
        patch.set_facecolor(cor_g[grupo_de[c]])
        patch.set_edgecolor(cor_g[grupo_de[c]])
    a1.set_yticks(range(1, len(cont) + 1), cont, fontsize=8)
    a1.axvline(0, color=TINTA_2, lw=0.6)
    a1.set_xlim(-1, 1)
    a1.set_title("Contínuas (caixa = P25–P75; haste = P1–P99)")

    dd = disc[::-1]
    pos = [(df[c] > 0).mean() * 100 for c in dd]
    neg = [(df[c] < 0).mean() * 100 for c in dd]
    a2.barh(range(len(dd)), pos, color=CLASSE_COR[1], height=0.7, label="+1")
    a2.barh(range(len(dd)), neg, left=pos, color=CLASSE_COR[-1], height=0.7,
            edgecolor="white", linewidth=1, label="−1")
    a2.set_yticks(range(len(dd)), dd, fontsize=8)
    a2.set_xlabel("% das janelas (último pregão)")
    a2.set_title("Binárias e eventos (% diferente de 0)")
    a2.legend(loc="lower right")
    for g, cor in cor_g.items():
        a1.plot([], [], color=cor, lw=6, label=g)
    a1.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.05), fontsize=8)
    fig.suptitle("Escala das 24 features no último pregão da janela "
                 f"(amostra de {len(df):,} janelas)".replace(",", "."),
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _salvar(fig, "fig09_features.png", pasta)


def main(ticker: str | None = None, cfg=CFG):
    _estilo()
    pasta = os.path.join(cfg.outputs_dir, "figuras")
    os.makedirs(pasta, exist_ok=True)

    data = load_dataset(cfg)
    panels, feat_names = build_panels(data, cfg)
    folds = build_folds(panels, cfg.split.cv_folds, cfg.data.forecast_horizon,
                        cfg.split.val_ratio)
    tk = ticker or cfg.focus_ticker
    if tk not in panels:
        raise SystemExit(f"{tk} não está entre os ativos válidos.")

    print(f"Gerando figuras ({tk}):")
    fig_precos_medias(data[tk], tk, pasta)
    fig_indicadores(data[tk], tk, pasta)
    fig_rotulo(data[tk], tk, pasta)
    fig_classes(panels, folds, pasta)
    fig_walkforward(panels, folds, pasta)
    fig_ativos_tempo(panels, folds, pasta)
    fig_limiar(data, panels, pasta)
    fig_janela(panels[tk], feat_names, pasta)
    fig_features(panels, feat_names, pasta)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
