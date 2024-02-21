import matplotlib.pyplot as plt
from utils import create_datasets,compile_and_fit


class Plots:
    def __init__(self, models, sequence_length, target_timestep, train_df, valid_df, test_df, epochs):
        self.val_performance = {}
        self.models = models
        self.model = models.lstm(32,32)
        self.sequence_length = sequence_length
        self.target_timestep = target_timestep

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.epochs = epochs

    def plot_batch_size(self):
        batch_sizes = [128, 256, 512, 1024]

        for batch_size in batch_sizes:
            train_ds, valid_ds, test_ds = create_datasets(train_df=self.train_df, valid_df=self.valid_df, test_df=self.test_df,
                                                      target_timestep=self.target_timestep, seq_length=self.sequence_length, batch_size=batch_size)

            compile_and_fit(self.models.lstm(32,32), self.epochs, train_ds=train_ds, valid_ds=valid_ds, patience=200)

            self.val_performance[batch_size] = self.model.evaluate(valid_ds)

        val_mae = [v[1] for v in self.val_performance.values()]

        plt.plot(batch_sizes, val_mae, label='Validation mae', marker='o', markersize=3)
        plt.xlabel('Batch size')
        plt.ylabel('Validation MAE')
        plt.show()

    def plot_learning_rate_size(self):
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.000001]

        for learning_rate in learning_rates:
            train_ds, valid_ds, test_ds = create_datasets(train_df=self.train_df, valid_df=self.valid_df, test_df=self.test_df,
                                                      target_timestep=self.target_timestep, seq_length=self.sequence_length, batch_size=1024)

            compile_and_fit(self.model, self.epochs, learning_rate=learning_rate, train_ds=train_ds, valid_ds=valid_ds, patience=200)

            self.val_performance[learning_rate] = self.model.evaluate(valid_ds)

        val_mae = [v[1] for v in self.val_performance.values()]

        plt.plot(learning_rates, val_mae, label='Validation mae', marker='o', markersize=3)
        plt.xlabel('Learning rate')
        plt.ylabel('Validation MAE')
        plt.show()

    def plot_units_size(self):
        units = [16, 32, 64, 128, 256, 512]

        for unit in units:
            train_ds, valid_ds, test_ds = create_datasets(train_df=self.train_df, valid_df=self.valid_df, test_df=self.test_df,
                                                      target_timestep=self.target_timestep, seq_length=self.sequence_length, batch_size=1024)

            model = self.models.lstm(rnn_units=unit,lstm_units=unit)
            compile_and_fit(model, self.epochs, learning_rate=0.001, train_ds=train_ds, valid_ds=valid_ds, patience=200)

            self.val_performance[unit] = model.evaluate(valid_ds)

        val_mae = [v[1] for v in self.val_performance.values()]

        plt.plot(units, val_mae, label='Validation mae', marker='o', markersize=3)
        plt.xlabel('Number of LSTM Units')
        plt.ylabel('Validation MAE')
        plt.show()

    def plot_sequence_size(self):
        sequence_sizes = [6, 8, 10, 12, 14, 16, 18, 20]

        for sequence_size in sequence_sizes:
            train_ds, valid_ds, test_ds = create_datasets(train_df=self.train_df, valid_df=self.valid_df, test_df=self.test_df,
                                                      target_timestep=self.target_timestep, seq_length=sequence_size, batch_size=1024)

            compile_and_fit(self.model, self.epochs, learning_rate=0.001, train_ds=train_ds, valid_ds=valid_ds, patience=200)

            self.val_performance[sequence_size] = self.model.evaluate(valid_ds)

        val_mae = [v[1] for v in self.val_performance.values()]

        plt.plot(units, val_mae, label='Validation mae', marker='o', markersize=3)
        plt.xlabel('History (Number of 5min intervals)')
        plt.ylabel('Validation MAE')
        plt.show()


