import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import compile_and_fit


class Models:
    """Builds, trains, and evaluates RNN architectures for glucose forecasting.

    All models predict a single glucose value `prediction_timestep` steps
    (30 minutes by default) into the future from a fixed-length input window.

    Args:
        seq_length: Number of input timesteps per window.
        train_ds: Training tf.data.Dataset.
        valid_ds: Validation tf.data.Dataset.
        test_ds: Test tf.data.Dataset.
        epochs: Maximum training epochs (early stopping will typically fire first).
    """

    # Number of 5-minute intervals to predict ahead (6 × 5 min = 30 min).
    PREDICTION_TIMESTEP = 6
    LEARNING_RATE = 0.001
    PATIENCE = 200

    def __init__(
        self,
        seq_length: int,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        epochs: int,
    ) -> None:
        self.seq_length = seq_length
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.epochs = epochs
        self.multi_val_performance: dict = {}
        self.multi_performance: dict = {}

    # ------------------------------------------------------------------
    # Model architectures
    # ------------------------------------------------------------------

    def lstm(self, lstm_units: int, rnn_units: int) -> tf.keras.Model:
        """LSTM layer followed by two SimpleRNN layers.

        The hybrid architecture retains long-term memory (LSTM) while keeping
        later layers lightweight (SimpleRNN).
        """
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1),
        ])

    def gru(self, gru_units: int, rnn_units: int) -> tf.keras.Model:
        """GRU layer followed by two SimpleRNN layers."""
        return tf.keras.Sequential([
            tf.keras.layers.GRU(gru_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1),
        ])

    def rnn(self, rnn_units: int) -> tf.keras.Model:
        """Stacked SimpleRNN baseline."""
        return tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1),
        ])

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def compile_and_fit_model(
        self, model: tf.keras.Model, patience: int = PATIENCE
    ) -> tf.keras.callbacks.History:
        """Compile and train a single model using the instance datasets."""
        return compile_and_fit(
            model,
            self.epochs,
            self.LEARNING_RATE,
            self.train_ds,
            self.valid_ds,
            patience=patience,
        )

    def compile_and_fit_all_models(self) -> tuple[dict, dict]:
        """Train LSTM, RNN, and GRU models and record validation/test MAE.

        Returns:
            Tuple of (val_performance, test_performance) dicts keyed by
            architecture name.
        """
        configs = {
            "LSTM": self.lstm(lstm_units=16, rnn_units=16),
            "RNN":  self.rnn(rnn_units=16),
            "GRU":  self.gru(gru_units=16, rnn_units=16),
        }
        for name, model in configs.items():
            compile_and_fit(
                model,
                self.epochs,
                learning_rate=self.LEARNING_RATE,
                train_ds=self.train_ds,
                valid_ds=self.valid_ds,
                patience=self.PATIENCE,
            )
            self.multi_val_performance[name] = model.evaluate(self.valid_ds, verbose=0)
            self.multi_performance[name] = model.evaluate(self.test_ds, verbose=0)

        return self.multi_val_performance, self.multi_performance

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_thirty_min_predictions(
        self,
        model: tf.keras.Model,
        test_df: pd.DataFrame,
        train_std: float,
        train_mean: float,
    ) -> None:
        """Plot 30-minute-ahead predictions against ground truth on the test set.

        Args:
            model: Trained Keras model.
            test_df: Normalised test DataFrame.
            train_std: Training set standard deviation (used to denormalise).
            train_mean: Training set mean (used to denormalise).
        """
        n_predictions = len(test_df) - self.seq_length
        y_pred = np.zeros(n_predictions)

        for i in range(n_predictions):
            window = test_df.to_numpy()[np.newaxis, i : self.seq_length + i]
            y_pred[i] = model.predict(window, verbose=0)[0]

        y_true = (test_df * train_std + train_mean).to_numpy()
        y_pred = y_pred * train_std + train_mean

        time_true = np.arange(len(y_true)) * 5
        pred_start = self.seq_length + self.PREDICTION_TIMESTEP - 1
        time_pred = np.arange(pred_start, pred_start + n_predictions) * 5

        plt.plot(time_true, y_true, label="Ground truth", linestyle="--", marker="o", markersize=3)
        plt.plot(time_pred, y_pred, label="Prediction (30 min ahead)", marker="o", markersize=2)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Blood glucose (mmol/L)")
        plt.title("30-Minute Blood Glucose Prediction")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_performance(self) -> None:
        """Train all architectures and plot a grouped bar chart of MAE scores."""
        val_performance, test_performance = self.compile_and_fit_all_models()

        x = np.arange(len(test_performance))
        width = 0.3
        val_mae = [v[1] for v in val_performance.values()]
        test_mae = [v[1] for v in test_performance.values()]

        plt.bar(x - 0.17, val_mae, width, label="Validation")
        plt.bar(x + 0.17, test_mae, width, label="Test")
        plt.xticks(ticks=x, labels=test_performance.keys(), rotation=45)
        plt.ylabel("MAE (mmol/L)")
        plt.title("Model Comparison — Validation vs Test MAE")
        plt.legend()
        plt.tight_layout()
        plt.show()
