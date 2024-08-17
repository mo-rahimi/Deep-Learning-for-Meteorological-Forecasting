from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

# In below each time one model is used
'''
model = load_model("lstm_2days_prediction.h5")
with open("history_2days.pkl", "rb") as file:
    loaded_history = pickle.load(file)
'''

'''
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
model = load_model("lstm_2days_prediction_try3.h5", custom_objects={"root_mean_squared_error": root_mean_squared_error})
with open("history_2days_try3.pkl", "rb") as file:
    loaded_history = pickle.load(file)
'''

model = load_model("lstm_2days_prediction_try4.h5")
with open("history_2days.pkl_try4", "rb") as file:
    loaded_history = pickle.load(file)

def plot_loss(loaded_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loaded_history['loss'], label='Training Loss')
    plt.plot(loaded_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs for 2 days prediction')
    plt.show()

plot_loss(loaded_history)
def plot_loss_log(loaded_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loaded_history['loss'], label='Training Loss')
    plt.plot(loaded_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')  # Seting the y-axis to logarithmic scale
    plt.legend()
    plt.title('Loss vs. Epochs for 2 days prediction (Logarithmic Scale)')
    plt.show()

    # Print the loss values at the last epoch
    last_epoch = len(loaded_history['loss'])
    print(f"Training loss at epoch {last_epoch}: {loaded_history['loss'][-1]}")
    print(f"Validation loss at epoch {last_epoch}: {loaded_history['val_loss'][-1]}")

plot_loss_log(loaded_history)



