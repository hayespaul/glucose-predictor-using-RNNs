from config import initialise_dataset
from utils import create_datasets
from models import Models
from plots import Plots

df = initialise_dataset()

n = len(df)
train_df = df[0:int(n * 0.7)]
valid_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

train_mean = train_df.mean()["glucose"]
train_std = train_df.std()["glucose"]

train_df = (train_df - train_mean) / train_std
valid_df = (valid_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

train_ds, valid_ds, test_ds = create_datasets(train_df=train_df, valid_df=valid_df, test_df=test_df, target_timestep=6, seq_length=12, batch_size=1024)

models = Models(seq_length=12, train_ds=train_ds,valid_ds=valid_ds,test_ds=test_ds, epochs=4)

plots = Plots(models, sequence_length=12, target_timestep=6, train_df=train_df,valid_df=valid_df,test_df=test_df,epochs=4)
plots.plot_units_size()
#
# models.plot_performance()
#
# lstm_model = models.lstm(lstm_units=32,rnn_units=32)
# history = models.compile_and_fit_model(lstm_model)
# models.plot_thirty_min_predictions(model=lstm_model,test_df=test_df,train_std=train_std,train_mean=train_mean)

