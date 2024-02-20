import tensorflow as tf
import numpy as np
from utils import compile_and_fit
import matplotlib.pyplot as plt


class Models:
    def __init__(self, seq_length, train_ds, valid_ds, test_ds):
        self.multi_val_performance = {}
        self.multi_performance = {}
        self.seq_length = seq_length

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.prediction_timestep = 6

    def lstm(self, lstm_units, rnn_units):
        multi_lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1)
        ])
        return multi_lstm_model

    def cnn(self, conv_width=3, conv_filters=32):
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(conv_filters, activation='relu', kernel_size=conv_width),
            tf.keras.layers.Dense(1),
        ])
        return conv_model

    def dense(self, units=32):
        dense_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dense(1),
        ])
        return dense_model

    def linear(self,units=32):
        linear_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units, kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape(1)
        ])
        return linear_model

    def baseline_last_value(self):
        class MultiStepLastBaseline(tf.keras.Model):
            def call(self, inputs):
                return tf.tile(inputs[:, -1:, :], [1, self.prediction_timestep, 1])
        return MultiStepLastBaseline

    def compile_and_fit_all_models(self):
        lstm_model = self.lstm(lstm_units=32,rnn_units=32)
        compile_and_fit(lstm_model, train_ds=self.train_ds, valid_ds=self.valid_ds, patience=200)

        self.multi_val_performance['LSTM'] = lstm_model.evaluate(self.valid_ds)
        self.multi_performance['LSTM'] = lstm_model.evaluate(self.test_ds, verbose=0)

        return self.multi_val_performance, self.multi_performance

    def compile_and_fit_model(self, model, patience=200):
        history = compile_and_fit(model, self.train_ds, self.valid_ds, patience=patience)
        return history

    def plot_thirty_min_predictions(self, model, test_df, train_std, train_mean):
        y_pred_30 = np.zeros(len(test_df.to_numpy()) - self.seq_length)
        for five_min_intervals in range(len(y_pred_30)):
            X = test_df.to_numpy()[np.newaxis, five_min_intervals:self.seq_length - 1 + five_min_intervals]
            all_y_preds = model.predict(X)
            y_pred_30[five_min_intervals] = all_y_preds[0] * train_std + train_mean

        y_true = test_df * train_std + train_mean
        y_true = y_true.to_numpy()
        time_true = np.arange(len(y_true)) * 5
        time_pred_30 = np.arange(self.seq_length + 6 - 1, len(y_pred_30) + 6 + self.seq_length - 1) * 5

        plt.plot(time_true, y_true, label='ground truth', linestyle='--', marker='o', markersize=3)
        plt.plot(time_pred_30, y_pred_30, label='prediction 30 mins', marker='o', markersize=2)
        plt.show()

    def plot_performance(self):
        x = np.arange(len(self.multi_performance))
        width = 0.3
        lstm_model = self.lstm(lstm_units=32,rnn_units=32)

        metric_index = lstm_model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in self.multi_val_performance.values()]
        test_mae = [v[metric_index] for v in self.multi_performance.values()]

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=self.multi_performance.keys(),rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        plt.legend()
        plt.show()


