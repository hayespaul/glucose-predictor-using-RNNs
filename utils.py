import tensorflow as tf
import pandas as pd


def compile_and_fit(
    model: tf.keras.Model,
    epochs: int,
    learning_rate: float,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    patience: int,
) -> tf.keras.callbacks.History:
    """Compile and train a Keras model with SGD and early stopping.

    Args:
        model: Uncompiled Keras model.
        epochs: Maximum number of training epochs.
        learning_rate: SGD learning rate.
        train_ds: Training dataset.
        valid_ds: Validation dataset used for early stopping.
        patience: Number of epochs with no improvement before stopping.

    Returns:
        Keras History object containing training metrics.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=optimizer,
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=[early_stopping],
    )
    return history


def create_datasets(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_timestep: int,
    seq_length: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create windowed TensorFlow datasets for sequence-to-point forecasting.

    Each input window of `seq_length` steps is paired with a target
    `target_timestep` steps after the end of the window.

    Args:
        train_df: Normalised training DataFrame.
        valid_df: Normalised validation DataFrame.
        test_df: Normalised test DataFrame.
        target_timestep: Number of steps ahead to predict (e.g. 6 → 30 min).
        seq_length: Number of input timesteps per window.
        batch_size: Batch size.

    Returns:
        Tuple of (train_ds, valid_ds, test_ds).
    """
    # Target offset: seq_length - 1 is the index of the last input step;
    # adding target_timestep gives the index of the label to predict.
    target_offset = seq_length + target_timestep - 1

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        train_df.to_numpy(),
        targets=train_df[target_offset:],
        sequence_length=seq_length,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        valid_df.to_numpy(),
        targets=valid_df[target_offset:],
        sequence_length=seq_length,
        batch_size=batch_size,
    )
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        test_df.to_numpy(),
        targets=test_df[target_offset:],
        sequence_length=seq_length,
        batch_size=batch_size,
    )
    return train_ds, valid_ds, test_ds
