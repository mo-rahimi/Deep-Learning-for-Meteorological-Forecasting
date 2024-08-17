import pandas as pd
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)  # Show all rows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


df = pd.read_csv("Weather_HK_00.csv")
#print(df.head())
# Handel the column Date
df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# We do not need the column date any more
df = df.iloc[:,1:]

# create new column "Seconds"
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)

# Convert column date to 2 new columns 'Year sin' and 'Year cos'
day  = 24*60*60
year = (365.2425)*day
df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))

# We do not need column "Seconds" any more
df = df.drop('Seconds', axis=1)



'''
# plot the `Year_sin` and `Year_cos` based on the column `Seconds`
plt.plot(np.array(df['Year sin'])[:1000])
plt.plot(np.array(df['Year cos'])[:1000])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.savefig('time_of_day_signal.png', format='png')
plt.show()
'''



# Handel the column Wind
mean_wind_speed = df.pop('Mean_Wind_Speed(km/h)')
# Convert to radians.
wind_direction_rad = df.pop('Prevailing_Wind_Direction')*np.pi / 180

# Calculate the max wind x and y components.
df['mean_wind_x'] = mean_wind_speed*np.cos(wind_direction_rad)
df['mean_wind_y'] = mean_wind_speed*np.sin(wind_direction_rad)

'''
plt.hist2d(df['mean_wind_x'], df['mean_wind_y'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [km/h]')
plt.ylabel('Wind Y [km/h]')
ax = plt.gca()
ax.axis('tight')
plt.savefig('wind_x_y.png', format='png')
plt.show()

'''
print("Shape of df is :", df.shape)
print(df[:3])



## Split the data set
#We'll use a (70%, 20%, 10%) split for the training, validation, and test sets.
train_size = 8766*0.7
evaluation_size = 8766*0.2
test_size = 8766*0.1


print("The number of data points for traing would be     :", train_size)
print("The number of data points for evaluating would be :", evaluation_size)
print("The number of data points for testing would be    :", test_size)


#
# 16 inputs, 3 outputs for the first day and 3 for the second day
# prediction for the next 2 days
def df_to_X_y_days(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size - 1):  # Subtract 1 to account for the extra day in prediction
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label_day_1 = [df_as_np[i + window_size][2], df_as_np[i + window_size][6],
                       df_as_np[i + window_size][11]]
        label_day_2 = [df_as_np[i + window_size + 1][2], df_as_np[i + window_size + 1][6],
                       df_as_np[i + window_size + 1][11]]
        label = label_day_1 + label_day_2  # Combine the labels for both days
        y.append(label)
    return np.array(X), np.array(y)
X_days, y_days = df_to_X_y_days(df)
X_days_train, y_days_train = X_days[:6000], y_days[:6000]
X_days_val, y_days_val = X_days[6000:7800], y_days[6000:7800]
X_days_test, y_days_test = X_days[7800:], y_days[7800:]


print ("The shape of X is : ", X_days.shape)
print ("The shape of y is : ", y_days.shape)

print ("\nThe shape of X_train is : ", X_days_train.shape)
print ("The shape of y_train is : ", y_days_train.shape)

print ("The shape of X_val is : ", X_days_val.shape)
print ("The shape of y_val is : ", y_days_val.shape)

print ("The shape of X_test is : ", X_days_test.shape)
print ("The shape of y_test is : ", y_days_test.shape)



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

# Below is just an object of Adam optimizer to try in other models
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


# LSTM model
def create_lstm_model(input_shape, num_outputs):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True , activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(num_outputs , activation='linear'))
    model.compile(optimizer= optimizer, loss='mse', metrics=['mae'])
    return model

input_shape = (X_days_train.shape[1], X_days_train.shape[2])
num_outputs = y_days_train.shape[1]
model = create_lstm_model(input_shape, num_outputs)

# Train the model
history = model.fit(
    X_days_train, y_days_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_days_val, y_days_val),
    verbose=1
)
model.save("lstm_2days_prediction_try2.h5")

import pickle
with open("history_2days_try2.pkl", "wb") as file:
    pickle.dump(history.history, file)