# Model architectures

Each architecture lives in its own module under `src/models/` and has
an accompanying builder function returning a compiled
`tensorflow.keras.Sequential`.

## Baseline (Section 3)

```
Input        : (look_back, 1)
LSTM         : 50 units, return_sequences=False
Dropout      : 0.2
Dense        : 1 unit (linear)
------------------------------
Optimiser    : Adam (default LR = 1e-3)
Loss         : MSE
Callbacks    : EarlyStopping(monitor="val_loss", patience=20)
```

Purpose: provide a reference error floor against which the fine-tuned
model is compared.

## Tuned (Section 4)

```
Input        : (look_back, 1)
LSTM         : 128 units, return_sequences=True
Dropout      : 0.2
LSTM         : 64 units
Dropout      : 0.2
Dense        : 25 units
Dense        : 1 unit (linear)
------------------------------
Optimiser    : Adam(lr=1e-3)
Loss         : MSE
Callbacks    : EarlyStopping + ModelCheckpoint("best_btc_model.keras")
```

Purpose: richer representation via stacked LSTM layers. Best-epoch
weights are checkpointed to disk and restored before evaluation.

## Final (Section 6)

```
Input        : (look_back, 1)
LSTM         : 100 units, return_sequences=False
Dropout      : 0.2
Dense        : 1 unit (linear)
------------------------------
Optimiser    : Adam
Loss         : MSE
Trained on   : merged (train + test) = full 2012-2014
Used for     : 2015 out-of-sample inference
```

Purpose: deployed model for Phase 2. Hyper-parameters
(`look_back=90, units=100`) were selected by the grid search.

## Grid search (Section 5)

A light exhaustive search over `look_back × neurons`:

| Hyper-parameter | Values |
| --------------- | ------ |
| `look_back` | 60, 90 |
| `neurons` | 50, 100 |
| `epochs` | 20 |
| `batch_size` | 32 |
| `patience` | 5 |

Criterion: MAE in the USD domain on the test slice.

## Notes on determinism

Training LSTMs deterministically on GPU requires the environment
variable `TF_DETERMINISTIC_OPS=1` in addition to the seeds set by
`src.utils.seed.set_global_seed`. CPU-only runs are reproducible out of
the box once seeded.
