import pandas as pd
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)  # Show all rows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed



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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def create_lstm_model(input_shape, lstm_units):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(1, input_shape[-1]))
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(input_shape[-1]))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Full model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_inputs, encoder_states, decoder_inputs, encoder_lstm, decoder_lstm, decoder_dense

input_shape = (7, 16)
lstm_units = 64
model, encoder_inputs, encoder_states, decoder_inputs, encoder_lstm, decoder_lstm, decoder_dense = create_lstm_model(input_shape, lstm_units)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    [X_days_train, np.zeros((X_days_train.shape[0], 1, input_shape[-1]))], y_days_train,
    epochs=5,
    batch_size=32,
    validation_data=([X_days_val, np.zeros((X_days_val.shape[0], 1, input_shape[-1]))], y_days_val),
    verbose=1
)

# Separate encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Separate decoder model
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Make predictions
def predict(input_data):
    encoder_states = encoder_model.predict(input_data)
    decoder_input = np.zeros((input_data.shape[0], 1, input_shape[-1]))
    decoder_states = encoder_states
    predictions = decoder_model.predict([decoder_input] + decoder_states)
    return predictions[0]

predictions = predict(X_days_test)