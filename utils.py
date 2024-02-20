import tensorflow as tf


MAX_EPOCHS = 2


def compile_and_fit(model, train_ds, valid_ds,patience):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(train_ds, epochs=MAX_EPOCHS,
                        validation_data=valid_ds,
                        callbacks=[early_stopping])
    return history


def create_datasets(train_df, valid_df, test_df,target_timestep, seq_length):
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        train_df.to_numpy(),
        targets=train_df[seq_length + target_timestep:],
        sequence_length=seq_length,
        batch_size=1024,
        shuffle=True,
        seed=42
    )

    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        valid_df.to_numpy(),
        targets=valid_df[seq_length + 5:],
        sequence_length=seq_length,
        batch_size=1024,
    )

    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        test_df.to_numpy(),
        targets=test_df[seq_length + 5:],
        sequence_length=seq_length,
        batch_size=1024,
    )

    return train_ds, valid_ds, test_ds

