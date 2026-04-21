# `models/`

Trained Keras artefacts are saved here by the training scripts:

| File | Produced by |
| ---- | ----------- |
| `baseline_lstm.keras` | `scripts/run_train_baseline.py` |
| `best_btc_model.keras` | `scripts/run_train_tuned.py` (best-epoch checkpoint) |
| `final_lstm.keras` | `scripts/run_train_final.py` |

All `.keras` and `.h5` files are tracked with Git LFS (see
`.gitattributes`). If you just cloned the repository, run:

```bash
git lfs install
git lfs pull
```

to materialise the weights. Alternatively, retrain from scratch with:

```bash
make pipeline
```
