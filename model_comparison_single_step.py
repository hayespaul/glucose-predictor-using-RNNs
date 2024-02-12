from baseline_model import Baseline
from window_generator import WindowGenerator
from utils import compile_and_fit
from config import initialise_dataset
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

baseline = Baseline(label_index=column_indices['glucose'])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(two_hour_window.val)
performance['Baseline'] = baseline.evaluate(two_hour_window.test, verbose=0)


# two_hour_window.plot(example_inputs, example_labels, train_mean, train_std, model=baseline)


################## DENSE ##################

# predict 5 mins in future with 120 mins of data
CONV_WIDTH = 24
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['glucose'], test_df=test_df, train_df=train_df, val_df=val_df)

multi_step_dense = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    tf.keras.layers.Reshape([1, -1]),
])

dense_history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

dense_example_inputs, dense_example_labels = conv_window.split_window(example_window)

conv_window.plot(dense_example_inputs, dense_example_labels, train_mean, train_std, model=multi_step_dense)

################## CONV ##################

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])


conv_history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['glucose'], test_df=test_df, train_df=train_df, val_df=val_df)

conv_example_window = tf.stack([np.array(train_df[:wide_conv_window.total_window_size]),
                                np.array(train_df[100:100 + wide_conv_window.total_window_size]),
                                np.array(train_df[200:200 + wide_conv_window.total_window_size])])

conv_example_inputs, conv_example_labels = wide_conv_window.split_window(conv_example_window)

wide_conv_window.plot(conv_example_inputs, conv_example_labels, train_mean, train_std, model=conv_model)


################## LSTM ##################

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(lstm_model, two_hour_window)

val_performance['LSTM'] = lstm_model.evaluate(two_hour_window.val)
performance['LSTM'] = lstm_model.evaluate(two_hour_window.test, verbose=0)

two_hour_window.plot(example_inputs, example_labels, train_mean, train_std, model=lstm_model)

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Glucose (mmol/l)]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
plt.legend()
plt.show()
