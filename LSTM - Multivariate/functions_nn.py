import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.regularizers import l2

def prepare_data(df, features, target):

    # Create the dataset with features and filter the data to the list of FEATURES
    df_features = df[features]
    
    # Convert the data to numpy values
    arr_use_unscaled = np.array(df_features)
    arr_target_unscaled = np.array(df[target]).reshape(-1, 1)
    
    return arr_use_unscaled, arr_target_unscaled, df_features

def partition_dataset(input_sequence_length, output_sequence_length, data, index_Close):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(input_sequence_length, data_len - output_sequence_length +1):
        x.append(data[i-input_sequence_length:i,:]) #contains input_sequence_length values 0-input_sequence_length * columns
        y.append(data[i:i + output_sequence_length, index_Close]) #contains the prediction values for validation (3rd column = Close),  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

def create_lstm_model(model_type, x_train, output_sequence_length, l2_reg=0.01):
    # Create a Sequential model
    model = Sequential()
    
    # Define the number of LSTM units
    n_lstm_units = 64  # You can adjust this number based on your problem
    
    if model_type == 'simple':
        # Simple LSTM model
        model.add(LSTM(n_lstm_units, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(5))
        model.add(Dense(output_sequence_length))
    
    elif model_type == 'complex':
        # Complex LSTM model without regularization
        n_input_neurons = x_train.shape[1] * x_train.shape[2]
        model.add(LSTM(n_input_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(n_input_neurons, return_sequences=False))
        model.add(Dense(5))
        model.add(Dense(output_sequence_length))
    
    elif model_type == 'complex_l2':
        # Complex LSTM model with L2 regularization
        n_input_neurons = x_train.shape[1] * x_train.shape[2]
        model.add(LSTM(n_input_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]),
                       kernel_regularizer=l2(l2_reg)))
        model.add(LSTM(n_input_neurons, return_sequences=False, kernel_regularizer=l2(l2_reg)))
        model.add(Dense(5, kernel_regularizer=l2(l2_reg)))
        model.add(Dense(output_sequence_length, kernel_regularizer=l2(l2_reg)))
    else:
        raise ValueError("Invalid model_type. Use 'simple' or 'complex'.")
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model