from config import initialise_dataset
from utils import create_datasets
from models import Models
from plots import Plots

# --- Hyperparameters ---
SEQ_LENGTH = 12       # 12 × 5 min = 1 hour of input history
TARGET_TIMESTEP = 6   # 6 × 5 min = 30 min forecast horizon
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
MAX_EPOCHS = 10_000   # Early stopping (patience=200) will fire well before this


def main() -> None:
    # --- Data loading and splitting (70 / 20 / 10) ---
    df = initialise_dataset()
    n = len(df)
    train_df = df[: int(n * 0.7)]
    valid_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df  = df[int(n * 0.9) :]

    # Normalise using training statistics only to prevent data leakage.
    train_mean = train_df.mean()["glucose"]
    train_std  = train_df.std()["glucose"]
    train_df = (train_df - train_mean) / train_std
    valid_df = (valid_df - train_mean) / train_std
    test_df  = (test_df  - train_mean) / train_std

    train_ds, valid_ds, test_ds = create_datasets(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        target_timestep=TARGET_TIMESTEP,
        seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # --- Hyperparameter sensitivity plots ---
    # Uses a separate epoch budget (1000) so sweeps finish in reasonable time.
    sweep_models = Models(
        seq_length=SEQ_LENGTH,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        epochs=1000,
    )
    plots = Plots(
        models=sweep_models,
        learning_rate=LEARNING_RATE,
        sequence_length=SEQ_LENGTH,
        target_timestep=TARGET_TIMESTEP,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        epochs=1000,
    )
    plots.plot_sequence_length()
    plots.plot_units()
    plots.plot_learning_rate()
    plots.plot_batch_size()

    # --- Final model training and evaluation ---
    final_models = Models(
        seq_length=SEQ_LENGTH,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        epochs=MAX_EPOCHS,
    )
    final_models.plot_performance()

    lstm_model = final_models.lstm(lstm_units=16, rnn_units=16)
    final_models.compile_and_fit_model(lstm_model)
    final_models.plot_thirty_min_predictions(lstm_model, test_df, train_std, train_mean)


if __name__ == "__main__":
    main()
