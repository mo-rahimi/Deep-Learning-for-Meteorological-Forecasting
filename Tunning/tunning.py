from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

model = load_model("lstm_2days_prediction.h5")
with open("history_2days.pkl", "rb") as file:
    loaded_history = pickle.load(file)

# Copy codes from LSTM_2days_prediction.py file to use the test data set for prediction, so please ignore lines 12 to 66
df = pd.read_csv("Weather_HK_00.csv")
df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.iloc[:,1:]
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
day  = 24*60*60
year = (365.2425)*day
df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
df = df.drop('Seconds', axis=1)
mean_wind_speed = df.pop('Mean_Wind_Speed(km/h)')
wind_direction_rad = df.pop('Prevailing_Wind_Direction')*np.pi / 180
df['mean_wind_x'] = mean_wind_speed*np.cos(wind_direction_rad)
df['mean_wind_y'] = mean_wind_speed*np.sin(wind_direction_rad)
def df_to_X_y_days(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size - 1):  # Subtract 1 to account for the extra day in prediction
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label_day_1 = [df_as_np[i + window_size][2], df_as_np[i + window_size][6], df_as_np[i + window_size][11]]
        label_day_2 = [df_as_np[i + window_size + 1][2], df_as_np[i + window_size + 1][6], df_as_np[i + window_size + 1][11]]
        label = label_day_1 + label_day_2  # Combine the labels for both days
        y.append(label)
    return np.array(X), np.array(y)
X_days, y_days = df_to_X_y_days(df)
X_days_train, y_days_train = X_days[:6000], y_days[:6000]
X_days_val, y_days_val = X_days[6000:7800], y_days[6000:7800]
X_days_test, y_days_test = X_days[7800:], y_days[7800:]
# Standardization
num_features = X_days_train.shape[-1]
training_mean = np.zeros(num_features)
training_std = np.zeros(num_features)
# Calculate mean and std for each feature
for i in range(num_features):
    training_mean[i] = np.mean(X_days_train[:, :, i])
    training_std[i] = np.std(X_days_train[:, :, i])
# Standardize input features
def preprocess(X):
    for i in range(num_features):
        if i not in [12, 13, 14, 15]:  # Skip Year_sin, Year_cos, mean_wind_x, mean_wind_y
            X[:, :, i] = (X[:, :, i] - training_mean[i]) / training_std[i]
    return X
# Standardize output features
def preprocess_output(y):
    output_feature_indices = [1, 6, 11]
    for idx, i in enumerate(output_feature_indices):
        y[:, idx] = (y[:, idx] - training_mean[i]) / training_std[i]
    return y
preprocess(X_days_train)
preprocess(X_days_val)
preprocess(X_days_test)
preprocess_output(y_days_train)
preprocess_output(y_days_val)
preprocess_output(y_days_test)



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from kerastuner import HyperModel, RandomSearch



class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, num_outputs):
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def build(self, hp):
        model = Sequential()
        model.add(
            LSTM(
                hp.Int("units_1", min_value=32, max_value=256, step=32),
                return_sequences=True,
                input_shape=self.input_shape,
                activation="tanh",
            )
        )
        model.add(Dropout(hp.Float("dropout_1", 0.1, 0.5, step=0.1)))
        model.add(
            LSTM(
                hp.Int("units_2", min_value=32, max_value=256, step=32),
                return_sequences=True,
                activation="tanh",
            )
        )
        model.add(Dropout(hp.Float("dropout_2", 0.1, 0.5, step=0.1)))
        model.add(
            LSTM(
                hp.Int("units_3", min_value=32, max_value=256, step=32),
                activation="tanh",
            )
        )
        model.add(Dropout(hp.Float("dropout_3", 0.1, 0.5, step=0.1)))
        model.add(Dense(self.num_outputs, activation="linear"))

        optimizer = Adam(
            learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        )
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

input_shape = X_days_train.shape[1:]
num_outputs = y_days_train.shape[1]
hypermodel = LSTMHyperModel(input_shape, num_outputs)

tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=30,
    executions_per_trial=2,
    directory="random_search",
    project_name="lstm_tuning",
)

tuner.search_space_summary()

tuner.search(
    x=X_days_train,
    y=y_days_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_days_val, y_days_val),
)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
