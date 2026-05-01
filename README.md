# transformer-prediction

GPT-style decoder-only Transformer for next-day price-move prediction on NSE daily OHLCV data. Adapted from nanoGPT — token embeddings replaced with a linear "indicator embedder" so the model consumes a sequence of numeric feature vectors instead of token IDs, and the final head regresses a single scalar target with MSE loss.

## Repo layout

```
NSE_D/                  raw daily OHLCV CSVs (Date,Open,High,Low,Close,Volume)
dataset/data/           generated feature CSVs (created by dimensions.py)
dimensions.py           feature engineering pipeline (raw -> dataset/data)
model.py                GPT model, IndicatorEmbedder, GPTConfig
train.py                training loop
config.py               legacy config dict (mostly unused by train.py)
config/model-1.py       legacy nanoGPT shakespeare config (unused)
valuation.py            plot rolling-average loss from losses.txt
README.md
```

## Data flow

1. `dimensions.py` reads each CSV in `NSE_D/`, computes features, writes the last 2520 rows to `dataset/data/<TICKER>.csv`.
2. `train.py` samples a random ticker file, slices a window of `interval` (=64) consecutive rows as input `x`, and uses the next row's first feature (`cl_op_t`) as target `y`.
3. The model sees a sequence of `interval` feature-vectors of width `ind_dim` (=8) and predicts a single scalar.

### Features computed by `dimensions.py`

Percent-change features (used by `train.py`):
- `cl_op_t`  = (Close − Open) / Open × 100
- `hi_op_t`  = (High − Open) / Open × 100
- `lo_op_t`  = (Low − Open) / Open × 100
- `op_cl_t_1` = (Open − Close[t−1]) / Close[t−1] × 100

Calendar features (normalised to roughly [0,1]):
- `Day` = day-of-month / 32
- `Week` = ISO week / 52
- `Month` = month / 12
- `Weekday` = weekday / 8

Volume: per-file min-max scaled to [0, 100].

`TechnicalIndicators` also implements EMA / RSI / MACD / Fibonacci pivots — these are written by older config but **not** used by `train.py`'s 8-dim feature set.

## Model

`model.py:GPT` — pre-norm Transformer block (LayerNorm → SelfAttention → residual → LayerNorm → MLP → residual).

- `IndicatorEmbedder`: stack of `nn.Linear` mapping `ind_dim → n_embd` (input projection in place of token embedding).
- Learned positional embedding `wpe` of length `block_size`.
- Causal self-attention via `F.scaled_dot_product_attention` (flash attention when available).
- `lm_head` projects `n_embd → pred_size` (default 1) and only the last time-step is used for the loss.
- Loss: `F.mse_loss(logits[:, -1, :], y)`.

Default `GPTConfig`:
```
block_size=64, ind_dim=8, pred_size=1, n_layer=6, n_head=8, n_embd=256, dropout=0.0, bias=True
```

## Usage

Requires Python 3.9+, PyTorch ≥ 2.0 (for flash attention), pandas, numpy, ta, matplotlib, tqdm.

```bash
# 1. build feature CSVs from raw NSE data
python dimensions.py

# 2. train
python train.py
```

`train.py` runs 100 outer × 10 inner = 1000 optimisation steps with `AdamW(lr=3e-4)` and plots the loss curve at the end.

## Status

Research / prototype. No checkpointing, no eval split, no CLI args — edit constants at the top of `train.py` to change behaviour. See "Known issues" below before running.

## Known issues

See the review notes accompanying this README — there are several bugs in `train.py` (variable shadowing, wrong loss-average divisor, dataset-file deletion on short series, broken target indexing) and dead/legacy files (`config.py`, `config/model-1.py`, unused DDP imports, debug `print` in `model.py:GPT.forward`).
