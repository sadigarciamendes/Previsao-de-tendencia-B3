#Parâmetros da etapa de coleta, tratamento e normalização dos dados da B3.

from dataclasses import dataclass, field
from typing import Dict, List


# Universo de ativos da B3 (sufixo .SA do yfinance).
TICKERS: List[str] = [
    "ABEV3.SA", "ALOS3.SA", "ALPA4.SA", "ANIM3.SA", "ARML3.SA",
    "ASAI3.SA", "AXIA3.SA", "AZZA3.SA", "B3SA3.SA", "BBAS3.SA",
    "BBDC3.SA", "BBDC4.SA", "BBSE3.SA", "BEEF3.SA", "BPAC11.SA",
    "BRAP4.SA", "BRAV3.SA", "BRFS3.SA", "BRKM5.SA", "CCRO3.SA",
    "CEAB3.SA", "CIEL3.SA", "CMIG4.SA", "CMIN3.SA", "COGN3.SA",
    "CPFE3.SA", "CPLE3.SA", "CRFB3.SA", "CSAN3.SA", "CSED3.SA",
    "CSMG3.SA", "CSNA3.SA", "CVCB3.SA", "CXSE3.SA", "CYRE3.SA",
    "DIRR3.SA", "EGIE3.SA", "EMBJ3.SA", "ENEV3.SA", "ENGI11.SA",
    "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA",
    "GUAR3.SA", "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "INTB3.SA",
    "ISAE4.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA",
    "LWSA3.SA", "MBRF3.SA", "MDIA3.SA", "MGLU3.SA", "MOTV3.SA",
    "MOVI3.SA", "MRVE3.SA", "MULT3.SA", "MYPK3.SA", "NATU3.SA",
    "ONCO3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "PETZ3.SA",
    "PGMN3.SA", "POMO4.SA", "PRIO3.SA", "PSSA3.SA", "RADL3.SA",
    "RAIL3.SA", "RAIZ4.SA", "RAPT4.SA", "RDOR3.SA", "RENT3.SA",
    "SANB11.SA", "SBSP3.SA", "SIMH3.SA", "SLCE3.SA", "SMTO3.SA",
    "STBP3.SA", "SUZB3.SA", "TAEE11.SA", "TEND3.SA", "TIMS3.SA",
    "TOTS3.SA", "TRIS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA",
    "VAMO3.SA", "VBBR3.SA", "VIVT3.SA", "WEGE3.SA", "YDUQ3.SA",
]

FOCUS_TICKER: str = "PETR4.SA"

# Ativos removidos por falha nos dados do yfinance (checagem manual do cache).
EXCLUDED: Dict[str, str] = {
    "NATU3.SA": "buraco de 2.017 dias (2019-2025, período como NTCO3) e queda de -72% na volta",
    "PCAR3.SA": "preço parado e volume zero em 2016-2020; salto de +139% em 01/02/2017",
    "UGPA3.SA": "salto de +119% em 28/06/2021 (desdobramento mal ajustado)",
    "SUZB3.SA": "preço parado e volume zero em 2016-2017 (limiar tau quase zero em 255 dias)",
    "TEND3.SA": "preço parado e volume zero em 2016-2017 (limiar tau quase zero em 119 dias)",
    "CPFE3.SA": "preço parado em 2017 (limiar tau quase zero em 79 dias)",
}


@dataclass
class DataConfig:
    period_years: int = 10
    interval: str = "1d"
    forecast_horizon: int = 10
    neutral_k: float = 0.5
    window_size: int = 30
    min_history: int = 400
    auto_adjust: bool = False
    rounding: bool = True
    actions: bool = True


@dataclass
class SplitConfig:
    """Parâmetros da divisão walk-forward (ver `walkforward.py`)."""
    cv_folds: int = 5            # blocos de teste out-of-sample
    val_ratio: float = 0.15      # cauda do treino de cada fold usada p/ early stopping


@dataclass
class DataPipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    tickers: List[str] = field(default_factory=lambda: list(TICKERS))
    excluded: Dict[str, str] = field(default_factory=lambda: dict(EXCLUDED))
    focus_ticker: str = FOCUS_TICKER
    outputs_dir: str = "outputs"


CFG = DataPipelineConfig()
