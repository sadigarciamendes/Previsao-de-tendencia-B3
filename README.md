# Processamento de Dados da B3 para Modelos de Tendência

Pipeline de processamento de dados de ações da B3: baixa o histórico, calcula
indicadores técnicos em forma de índices de força limitados a `[-1, 1]`, rotula a
tendência de curto prazo e organiza tudo em tensores de janelas rolantes prontos
para alimentar uma rede sequencial (ex.: LSTM).

> Este projeto cobre apenas a etapa de dados. Modelagem, treino e avaliação ficam
> fora do escopo.

## Fluxo

```
config.py
   |
   v
data_collection.py  ->  features.py  ->  dataset.py
 (OHLCV via yfinance)   (indicadores      (rotulagem + janelas
                         em força [-1,1])   + tensores por grupo)
```

Orquestrado por `process_data.py`.

## Componentes

| Módulo | Papel |
|---|---|
| `config.py` | Parâmetros (100 tickers, janela=30, horizonte=10, neutral_k=0.5, hold-out=20%, mínimo de 400 pregões). |
| `data_collection.py` | Baixa 10 anos de candles diários via yfinance (`auto_adjust=False`, `actions=True`, `rounding=True`), com limpeza e seleção. |
| `features.py` | Indicadores via TA-Lib (com fallback em pandas) reescritos como índices de força em [-1,1]: RSI(14), MACD(12,26,9), SMA(50/200)+EMA(9/20), Bollinger(20,2) e OBV/Volume. |
| `dataset.py` | Rotulagem em 3 classes (-1=baixa, 0=neutra, +1=alta) com banda neutra escalada pela volatilidade, janela rolante de 30, sem normalizador ajustado, hold-out final por ativo e pooling. |
| `process_data.py` | Executa o pipeline ponta a ponta e salva um resumo. |
| `export_csv.py` | Exporta o dataframe processado de um ativo (`close` + 24 atributos + `label` -1/0/+1) para CSV. |

## Normalização por construção

Não há `StandardScaler` nem `MinMaxScaler`. Cada indicador é escrito como um índice
de força que já nasce em `[-1, 1]`, por exemplo, para médias móveis:

```
forca = (media_curta - media_longa) / (media_curta + media_longa)
```

Isso mantém a entrada estável quando o modelo encontra um período ou um ativo com
patamar de preço diferente do observado no treino, sem depender de parâmetros
estimados a partir dos dados de treino.

## Como executar

```bash
# 1) (recomendado) ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2) dependências
pip install -r requirements.txt

# 3) rodar o pipeline de dados
python process_data.py

# 4) (opcional) exportar o dataframe de um ativo para CSV
python export_csv.py            # usa o focus_ticker (PETR4.SA)
python export_csv.py VALE3.SA   # qualquer ticker da B3
```

O `process_data.py` salva um resumo em `outputs/` com, por grupo de indicador, o
formato dos tensores de treino e de hold-out, os nomes das features e o intervalo
observado dos valores (que deve permanecer dentro de `[-1, 1]`).

## Saída de `build_supervised`

`build_supervised(data, cfg)` retorna:

- `tensors_by_group`: por grupo, `X_train/y_train` (pool de treino de todos os
  ativos) e `X_test/y_test` (hold-out agregado).
- `cv_by_group`: janelas de treino separadas por ativo, prontas para uma validação
  cruzada com avanço temporal (walk-forward).
- `holdout`: recorte de hold-out do ativo de foco (`focus_ticker`).
- `slices`: recorte de hold-out de cada ativo.

## Notas

- TA-Lib é opcional. Sem ele, `features.py` usa um cálculo equivalente em pandas
  com os mesmos parâmetros.
- Os indicadores são calculados sobre o preço de fechamento, sem transformação
  prévia de escala dos preços, pois a normalização é feita pela própria construção
  dos índices de força.
