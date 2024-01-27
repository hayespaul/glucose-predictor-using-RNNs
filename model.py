import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


df = pd.read_csv('//export_HK/cgm_data.csv', parse_dates=["Timestamp"])
df.columns = ["time", "glucose",'Serial number']
df = df[df['glucose'] <= 40]
df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M')

cutoff_time = pd.to_datetime('13/12/2023 07:00', format='%d/%m/%Y %H:%M')
df = df[~((df['Serial number'] == 'Abbott LibreView CSV (456F4EF)') & (df['time'] > cutoff_time))]
df = df.drop(columns=['Serial number'])
df = df.sort_values("time").set_index("time")
# print(df.head())

# plt.plot(df['time'], df['glucose'])
# plt.show()
# 21/12/2023 00:01,9.8,1183798
# 20/12/2023 23:56,9.2,1183798
# valid_end_date = '2023-12-29 23:57:00'
train_end_date = '2023-12-28 23:56:00'

# train_end_date = '2023-12-27 23:56:00'
train_df = df.loc[:train_end_date] / 10

valid_start_date = '2023-12-29 00:01:00'
valid_end_date = '2023-12-29 23:57:00'
valid_df = df.loc[valid_start_date:] / 10

# test_start_date = '2023-12-30 00:02:00'
# test_df = df.loc[test_start_date:] / 10

test_df = valid_df

seq_length = 8

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    train_df.to_numpy(),
    targets=train_df[seq_length+5:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    valid_df.to_numpy(),
    targets=valid_df[seq_length+5:],
    sequence_length=seq_length,
    batch_size=32,
)


test_ds = tf.keras.utils.timeseries_dataset_from_array(
    test_df.to_numpy(),
    targets=test_df[seq_length+5:],
    sequence_length=seq_length,
    batch_size=32,
)


def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))


def to_seq2seq_dataset(series, seq_length=12, ahead=6, batch_size=32, shuffle=False, seed=None):
    ds = to_windows(to_windows(tf.data.Dataset.from_tensor_slices(series), ahead+1),seq_length)
    ds = ds.map(lambda S: (S[:,0], S[:,1:]))
    if shuffle:
        ds = ds.shuffle(8*batch_size, seed=seed)
    return ds.batch(batch_size)


seq2seq_train = to_seq2seq_dataset(train_df, shuffle=True, seed=42)
seq2seq_valid = to_seq2seq_dataset(valid_df, shuffle=True, seed=42)


tf.random.set_seed(42)
univar_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(16,return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(16),
    tf.keras.layers.Dense(1)
])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=200, restore_best_weights=True)
opt = tf.keras.optimizers.SGD(learning_rate=0.09,momentum=0.9)
univar_model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
history = univar_model.fit(train_ds,validation_data=valid_ds, epochs=500,callbacks=[early_stopping_cb])

# seq2seq_model = tf.keras.Sequential([
#     tf.keras.layers.SimpleRNN(128,return_sequences=True, input_shape=[None, 1]),
#     tf.keras.layers.SimpleRNN(64,return_sequences=True),
#     tf.keras.layers.Dense(6)
# ])
#
# early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=200, restore_best_weights=True)
# opt = tf.keras.optimizers.SGD(learning_rate=0.09,momentum=0.9)
# seq2seq_model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
# history = seq2seq_model.fit(seq2seq_train,validation_data=seq2seq_valid, epochs=1000,callbacks=[early_stopping_cb])


# y_pred_5 = np.zeros(len(test_df.to_numpy()))
y_pred_30 = np.zeros(len(test_df.to_numpy())-seq_length)
# y_pred_60 = np.zeros(len(test_df.to_numpy()))
for five_min_intervals in range(len(y_pred_30)):
    X = test_df.to_numpy()[np.newaxis, five_min_intervals:seq_length-1+five_min_intervals]
    all_y_preds = univar_model.predict(X) * 10
    y_pred_30[five_min_intervals] = all_y_preds[0]


y_true = test_df.to_numpy()*10
time_true = np.arange(len(y_true))*5
time_pred_30 = np.arange(seq_length+6-1,len(y_pred_30)+6+seq_length-1)*5
plt.plot(time_true,y_true, label = 'ground truth', linestyle='--', marker='o', markersize=3)
# plt.plot(time_pred_5,y_pred_5[:-(seq_length)], label = 'prediction 5 mins')
plt.plot(time_pred_30,y_pred_30, label = 'prediction 30 mins', marker='o', markersize=2)
# plt.plot(time_pred_60,y_pred_60[:-(seq_length)], label = 'prediction 60 mins')
plt.xlabel("Time [mins]")
plt.ylabel("Glucose Concentration [mmol/L]")
plt.legend(loc='upper right')
plt.show()


#
# # y_pred_5 = np.zeros(len(test_df.to_numpy()))
# y_pred_30 = np.zeros(len(test_df.to_numpy()))
# # y_pred_60 = np.zeros(len(test_df.to_numpy()))
# for five_min_intervals in range(len(test_df.to_numpy())):
#     X = test_df.to_numpy()[np.newaxis, five_min_intervals:seq_length+five_min_intervals]
#     all_y_preds = seq2seq_model.predict(X)[0,-1] * 10
#     # y_pred_5[five_min_intervals] = all_y_preds[0]
#     y_pred_30[five_min_intervals] = all_y_preds[-1]
#     # y_pred_60[five_min_intervals] = all_y_preds[-1]
#
# y_true = test_df.to_numpy()*10
# time_true = np.arange(len(y_true))*5
# # time_pred_5 = np.arange(seq_length+1,len(y_pred_5)+1)*5
# time_pred_30 = np.arange(seq_length+6,len(y_pred_30)+6)*5
# # time_pred_60 = np.arange(seq_length+12,len(y_pred_5)+12)*5
# plt.plot(time_true,y_true, label = 'ground truth', linestyle='--')
# # plt.plot(time_pred_5,y_pred_5[:-(seq_length)], label = 'prediction 5 mins')
# plt.plot(time_pred_30,y_pred_30[:-(seq_length)], label = 'prediction 30 mins')
# # plt.plot(time_pred_60,y_pred_60[:-(seq_length)], label = 'prediction 60 mins')
# plt.xlabel("Time [mins]")
# plt.ylabel("Glucose Concentration [mmol/L]")
# plt.legend(loc='upper right')
# plt.show()

