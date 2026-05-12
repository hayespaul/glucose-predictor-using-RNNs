# Glucose Predictor

Predict blood glucose levels 30 minutes ahead using recurrent neural networks
(LSTM / GRU / SimpleRNN) trained on Continuous Glucose Monitor (CGM) data.

The project was built around 3 months of readings from a Dexcom G6 sensor
sampled every 5 minutes, but it works with any single-column CGM time series.

## What's in the repo

| File | Purpose |
| --- | --- |
| `run.py` | Entry point — loads data, trains the final model, produces all plots. |
| `config.py` | Loads and pre-processes the glucose CSV. |
| `utils.py` | Builds the windowed `tf.data.Dataset`s and the training loop. |
| `models.py` | LSTM / GRU / SimpleRNN architectures and evaluation plots. |
| `plots.py` | Hyperparameter sensitivity sweeps (units, learning rate, batch size, window length). |
| `glucose-predictor.ipynb` | Notebook write-up with results and discussion. |

## Setup

```bash
pip install tensorflow pandas numpy matplotlib
```

## Data

Place a CSV named `glucose.csv` in the project root with a single `glucose`
column containing readings in mmol/L at 5-minute intervals. To point at a
different file, edit `DATA_PATH` at the top of `config.py`.

Pre-processing (done automatically):
- Readings above 40 mmol/L are dropped as sensor errors.
- Missing readings are forward-filled with the previous timestep's value.

## Run

```bash
python run.py
```

This will:
1. Split the data 70 / 20 / 10 into train / validation / test sets.
2. Normalise using the training set's mean and std (no leakage from val/test).
3. Sweep hyperparameters and plot validation MAE for each.
4. Train the final LSTM + RNN model and plot 30-minute-ahead predictions
   against ground truth.

## Final model

The chosen configuration (selected from the sweeps):

- 60 minutes of history (12 × 5-minute steps)
- LSTM(16) → SimpleRNN(16) → SimpleRNN(16) → Dense(1)
- SGD with learning rate 0.001, momentum 0.9, batch size 1024
- Early stopping on validation loss (patience 200)

Reported test MAE: **~0.41 mmol/L** (mean over 100 runs, σ ≈ 0.02).
