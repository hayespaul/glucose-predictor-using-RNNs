from window_generator import WindowGenerator
from config import initialise_dataset
from utils import compile_and_fit
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

df = initialise_dataset()

n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

num_features = df.shape[1]

two_hour_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['glucose'], test_df=test_df, train_df=train_df, val_df=val_df)

example_window = tf.stack([np.array(train_df[:two_hour_window.total_window_size]),
                           np.array(train_df[100:100 + two_hour_window.total_window_size]),
                           np.array(train_df[200:200 + two_hour_window.total_window_size])])

example_inputs, example_labels = two_hour_window.split_window(example_window)

column_indices = {name: i for i, name in enumerate(df.columns)}


class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])


OUT_STEPS = 6
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS, test_df=test_df, train_df=train_df, val_df=val_df)


example_window_multi = tf.stack([np.array(train_df[:multi_window.total_window_size]),
                                np.array(train_df[100:100 + multi_window.total_window_size]),
                                np.array(train_df[200:200 + multi_window.total_window_size])])

example_inputs_multi, example_labels_multi = multi_window.split_window(example_window_multi)

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])


multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(example_inputs_multi, example_labels_multi, train_mean, train_std, model=last_baseline)


################################################################## multi linear model ##################################################################


multi_linear_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(example_inputs_multi, example_labels_multi, train_mean, train_std, model=multi_linear_model)


################################################################## multi dense model ##################################################################

multi_dense_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

dense_history = compile_and_fit(multi_dense_model, multi_window)

multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(example_inputs_multi, example_labels_multi, train_mean, train_std, model=multi_dense_model)

################################################################## multi CNN model ##################################################################


CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

conv_history = compile_and_fit(multi_conv_model, multi_window)


multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(example_inputs_multi, example_labels_multi, train_mean, train_std, model=multi_conv_model)

################################################################## multi LSTM model ##################################################################


multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

lstm_history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(example_inputs_multi, example_labels_multi, train_mean, train_std, model=multi_lstm_model)

################################################################## perfomance ##################################################################

x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
plt.legend()
plt.show()
