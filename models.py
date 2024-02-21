import tensorflow as tf
import numpy as np
from utils import compile_and_fit
import matplotlib.pyplot as plt


class Models:
    def __init__(self, seq_length, train_ds, valid_ds, test_ds, epochs):
        self.multi_val_performance = {}
        self.multi_performance = {}
        self.seq_length = seq_length

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.prediction_timestep = 6
        self.epochs = epochs

    def lstm(self, lstm_units, rnn_units):
        multi_lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1)
        ])
        return multi_lstm_model

    def GRU(self, GRU_units, rnn_units):
        GRU_model = tf.keras.Sequential([
            tf.keras.layers.GRU(GRU_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1)
        ])
        return GRU_model

    def RNN(self, rnn_units):
        RNN_model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(rnn_units),
            tf.keras.layers.Dense(1)
        ])
        return RNN_model

    def compile_and_fit_all_models(self):
        lstm_model = self.lstm(lstm_units=32,rnn_units=32)
        rnn_model = self.RNN(rnn_units=32)
        GRU_model = self.GRU(GRU_units=32,rnn_units=32)

        compile_and_fit(lstm_model, self.epochs, train_ds=self.train_ds, valid_ds=self.valid_ds, patience=200, )
        compile_and_fit(rnn_model,self.epochs, train_ds=self.train_ds, valid_ds=self.valid_ds, patience=200)
        compile_and_fit(GRU_model, self.epochs,train_ds=self.train_ds, valid_ds=self.valid_ds, patience=200)

        self.multi_val_performance['LSTM'] = lstm_model.evaluate(self.valid_ds)
        self.multi_performance['LSTM'] = lstm_model.evaluate(self.test_ds, verbose=0)

        self.multi_val_performance['RNN'] = rnn_model.evaluate(self.valid_ds)
        self.multi_performance['RNN'] = rnn_model.evaluate(self.test_ds, verbose=0)

        self.multi_val_performance['GRU'] = GRU_model.evaluate(self.valid_ds)
        self.multi_performance['GRU'] = GRU_model.evaluate(self.test_ds, verbose=0)

        return self.multi_val_performance, self.multi_performance

    def compile_and_fit_model(self, model, patience=200):
        history = compile_and_fit(model,self.epochs, self.train_ds, self.valid_ds, patience=patience)

        return history

    def plot_thirty_min_predictions(self, model, test_df, train_std, train_mean):
        y_pred_30 = np.zeros(len(test_df.to_numpy()) - self.seq_length)

        for five_min_intervals in range(len(y_pred_30)):
            X = test_df.to_numpy()[np.newaxis, five_min_intervals:self.seq_length - 1 + five_min_intervals]
            all_y_preds = model.predict(X)
            y_pred_30[five_min_intervals] = all_y_preds[0]

        y_true = test_df * train_std + train_mean
        y_true = y_true.to_numpy()
        time_true = np.arange(len(y_true)) * 5
        time_pred_30 = np.arange(self.seq_length + 6 - 1, len(y_pred_30) + 6 + self.seq_length - 1) * 5

        y_pred_30 = y_pred_30 * train_std + train_mean
        plt.plot(time_true, y_true, label='ground truth', linestyle='--', marker='o', markersize=3)
        plt.plot(time_pred_30, y_pred_30, label='prediction 30 mins', marker='o', markersize=2)
        plt.show()

    def plot_performance(self):
        multi_val_performance, multi_performance = self.compile_and_fit_all_models()
        x = np.arange(len(multi_performance))
        width = 0.3

        val_mae = [v[1] for v in multi_val_performance.values()]
        test_mae = [v[1] for v in multi_performance.values()]

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=multi_performance.keys(),rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        plt.legend()
        plt.show()
