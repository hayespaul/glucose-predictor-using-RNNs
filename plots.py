import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from utils import create_datasets, compile_and_fit
from models import Models


class Plots:
    """Generates hyperparameter sensitivity plots for the glucose prediction models.

    Each plot method sweeps one hyperparameter, trains a fresh model per value,
    and records validation MAE — allowing the optimal setting to be selected
    before final training.

    Args:
        models: Models instance used to construct architectures.
        learning_rate: Default learning rate for sweeps that don't vary it.
        sequence_length: Default input window length (number of 5-min steps).
        target_timestep: Forecast horizon in timesteps (e.g. 6 → 30 min).
        train_df: Normalised training DataFrame.
        valid_df: Normalised validation DataFrame.
        test_df: Normalised test DataFrame.
        epochs: Maximum epochs per training run (early stopping applies).
    """

    def __init__(
        self,
        models: Models,
        learning_rate: float,
        sequence_length: int,
        target_timestep: int,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        epochs: int,
    ) -> None:
        self.models = models
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.target_timestep = target_timestep
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.epochs = epochs

    def _make_datasets(self, seq_length: int, batch_size: int) -> tuple:
        return create_datasets(
            train_df=self.train_df,
            valid_df=self.valid_df,
            test_df=self.test_df,
            target_timestep=self.target_timestep,
            seq_length=seq_length,
            batch_size=batch_size,
        )

    def _train_and_evaluate(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        learning_rate: float,
        patience: int = 200,
    ) -> float:
        """Train a model and return its validation MAE."""
        compile_and_fit(model, self.epochs, learning_rate, train_ds, valid_ds, patience)
        return model.evaluate(valid_ds, verbose=0)[1]

    def plot_batch_size(self) -> None:
        """Plot validation MAE across a range of batch sizes."""
        batch_sizes = [128, 256, 512, 1024]
        val_mae = []

        for batch_size in batch_sizes:
            train_ds, valid_ds, _ = self._make_datasets(self.sequence_length, batch_size)
            model = self.models.lstm(lstm_units=256, rnn_units=256)
            val_mae.append(self._train_and_evaluate(model, train_ds, valid_ds, self.learning_rate))

        plt.plot(batch_sizes, val_mae, marker="o", markersize=4)
        plt.xlabel("Batch size")
        plt.ylabel("Validation MAE (mmol/L)")
        plt.title("Effect of Batch Size on Validation MAE")
        plt.tight_layout()
        plt.show()

    def plot_learning_rate(self) -> None:
        """Plot validation MAE across a range of learning rates."""
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        val_mae = []

        for lr in learning_rates:
            train_ds, valid_ds, _ = self._make_datasets(self.sequence_length, batch_size=1024)
            model = self.models.lstm(lstm_units=256, rnn_units=256)
            val_mae.append(self._train_and_evaluate(model, train_ds, valid_ds, lr))

        plt.plot(learning_rates, val_mae, marker="o", markersize=4)
        plt.xlabel("Learning rate")
        plt.ylabel("Validation MAE (mmol/L)")
        plt.title("Effect of Learning Rate on Validation MAE")
        plt.tight_layout()
        plt.show()

    def plot_units(self) -> None:
        """Plot validation MAE across a range of LSTM/RNN unit counts."""
        unit_counts = [16, 32, 64, 128, 256, 512]
        val_mae = []

        for units in unit_counts:
            train_ds, valid_ds, _ = self._make_datasets(self.sequence_length, batch_size=1024)
            model = self.models.lstm(lstm_units=units, rnn_units=units)
            # Higher patience here to ensure larger models have enough time to converge.
            val_mae.append(self._train_and_evaluate(model, train_ds, valid_ds, self.learning_rate, patience=1000))

        plt.plot(unit_counts, val_mae, marker="o", markersize=4)
        plt.xlabel("Number of units (LSTM and RNN)")
        plt.ylabel("Validation MAE (mmol/L)")
        plt.title("Effect of Network Size on Validation MAE")
        plt.tight_layout()
        plt.show()

    def plot_sequence_length(self) -> None:
        """Plot validation MAE across a range of input window lengths."""
        sequence_sizes = [6, 8, 10, 12, 14, 16, 18, 20]
        val_mae = []

        for seq_size in sequence_sizes:
            train_ds, valid_ds, _ = self._make_datasets(seq_size, batch_size=1024)
            model = self.models.lstm(lstm_units=16, rnn_units=16)
            val_mae.append(self._train_and_evaluate(model, train_ds, valid_ds, self.learning_rate))

        plt.plot(sequence_sizes, val_mae, marker="o", markersize=4)
        plt.xlabel("History length (number of 5-min intervals)")
        plt.ylabel("Validation MAE (mmol/L)")
        plt.title("Effect of Input Sequence Length on Validation MAE")
        plt.tight_layout()
        plt.show()
