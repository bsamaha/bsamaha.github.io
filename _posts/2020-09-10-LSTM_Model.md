---
title: "Project: Part 5, Time Series LSTM Forecast with Keras"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Deep Learning
  - Keras
  - LSTM
  - TensorFlow
---
**This notebook is part 5 of a multi-phase project of building a quantitative trading algorithm. This notebook is about LSTM models and how they operate.**


# Building an LSTM model to predict the price of the S&P 500

- LSTM stands for Long Short Term Memory

The LSTM cell contains an RNN cell inside (dense layer with a tanh activation function), but it has other attributes that allow it to remember patterns over a longer period of time. The LSTM contains a short-term state vector that is used from one time step to the next identical to the RNN cell.

Where LSTM cells differ is their Long-Term State vector. This long term state vector undergoes a multipicative and additive operation at each time step.The long term state vector starts off with a "forget gate" which is simply a sigmoid activation on top of a dense layer which produces an output from 0 - 1. This means if the forget gate ourputs a value of 1 then the long term state vector remains unchanged. On the opposite end, if the output valuse of the dense layer is 0 the long term state vector is nullified. This helps adjust quickly to highly volatile environments. If we are in a steady uptrend and then we suddenly have a big drop in the market, the long term state vector will be erased so the model can readjust quickly.

The next gate is the input gate. The input gate is much like the forget gate, except it decides if the short term state vector gets added to the long term vector or is erased. 

The last gate called the "output gate" decides what the next hidden state should be. It is important to know that this hidden state output is the prediction for the next time step. Here, the previous hidden state (previous cell output) amd current input are jammed together through sigmoid function. Then we take our newly modified long term state vector which has already passed through the forget and input gates, and apply a tanh function to it. Finally, multiple both the tanh output (long term state vector output) and the sigmoid function output (previous hidden state and new input at this time step) to what information the hidden state needs to carry to the next time step.

</br>
<b>To review, the Forget gate decides what is relevant to keep from prior steps. The input gate decides what information is relevant to add from the current step. The output gate determines what the next hidden state should be.</b>

## Setup


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

keras = tf.keras

# set style of charts
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = [10, 10]
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    

def sequential_window_dataset(series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)
```


```python
# Read in data
spy = pd.read_csv('SPY.csv')

# Convert series into datetime type
spy['Date'] = pd.to_datetime(spy['Date'])

# Save target series
series = spy['Close']

# Create train data set
train_split_date = '2014-12-31'
train_split_index = np.where(spy.Date == train_split_date)[0][0]
x_train = spy.loc[spy['Date'] <= train_split_date]['Close']

# Create test data set
test_split_date = '2019-01-02'
test_split_index = np.where(spy.Date == test_split_date)[0][0]
x_test = spy.loc[spy['Date'] >= test_split_date]['Close']

# Create valid data set
valid_split_index = (train_split_index.max(),test_split_index.min())
x_valid = spy.loc[(spy['Date'] < test_split_date) & (spy['Date'] > train_split_date)]['Close']
```


```python
# Plot all lines on one chart to see where one segment starts and another ends
plt.plot(x_train, label = 'Train')
plt.plot(x_valid, label = 'Validate')
plt.plot(x_test, label = 'Test')
plt.legend()
print(x_train.index.max(),x_valid.index.min(),x_valid.index.max(),x_test.index.min(),x_test.index.max())
```

    5521 5522 6527 6528 6949
    


![png](/assets/images/LSTM_Model/output_5_1.png)



```python
# Reshape values
x_train_values = x_train.values.reshape(-1, 1)
x_valid_values = x_valid.values.reshape(-1, 1)
x_test_values = x_test.values.reshape(-1, 1)

#  Create Scaler Object
x_train_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit x_train values
normalized_x_train = x_train_scaler.fit_transform(x_train_values)

# Fit x_valid values
normalized_x_valid = x_train_scaler.transform(x_valid_values)

# Fit x_test values
normalized_x_test = x_train_scaler.transform(x_test_values)

# All values normalized to training data
spy_normalized_to_traindata = x_train_scaler.transform(series.values.reshape(-1, 1))

# Example of how to iverse
# inversed = scaler.inverse_transform(normalized_x_train).flatten()
```


```python
class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()
```

## Find the learning Rate


```python
# reset any stored data
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Set window size and create input batch sequences
window_size = 20
train_set = sequential_window_dataset(normalized_x_train, window_size)

# create model
model = keras.models.Sequential([
  keras.layers.LSTM(100, return_sequences=True, stateful=True,
                    batch_input_shape=[1, None, 1]),
  keras.layers.LSTM(100, return_sequences=True, stateful=True),
  keras.layers.Dense(1),
])

# create lr
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-5 * 10**(epoch / 20))
reset_states = ResetStatesCallback()

# choose optimizer
optimizer = keras.optimizers.Nadam(lr=1e-5)

# compile model
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# set history
history = model.fit(train_set, epochs=100,
                    callbacks=[lr_schedule, reset_states])
```

    Epoch 1/100
    276/276 [==============================] - 3s 9ms/step - loss: 0.0109 - mae: 0.1198
    Epoch 2/100
    276/276 [==============================] - 2s 8ms/step - loss: 7.1192e-04 - mae: 0.0270
    Epoch 3/100
    276/276 [==============================] - 2s 8ms/step - loss: 4.6617e-04 - mae: 0.0222
... truncated

    Epoch 98/100
    276/276 [==============================] - 2s 8ms/step - loss: 0.3448 - mae: 0.7910
    Epoch 99/100
    276/276 [==============================] - 2s 7ms/step - loss: 0.4008 - mae: 0.8490
    Epoch 100/100
    276/276 [==============================] - 2s 7ms/step - loss: 0.5534 - mae: 1.0528
    


```python
# Plot the learning rate chart
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1, 0, 0.01])
```




    (1e-08, 1.0, 0.0, 0.01)




![png](/assets/images/LSTM_Model/output_10_1.png)


## Build and Train LSTM Model


```python
# reset any stored data
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# set window size and create input batch sequence
window_size = 30
train_set = sequential_window_dataset(normalized_x_train, window_size)
valid_set = sequential_window_dataset(normalized_x_valid, window_size)

# create model
model = keras.models.Sequential([
  keras.layers.LSTM(100, return_sequences=True, stateful=True,
                         batch_input_shape=[1, None, 1]),
  keras.layers.LSTM(100, return_sequences=True, stateful=True),
  keras.layers.Dense(1),
])

# set optimizer
optimizer = keras.optimizers.Nadam(lr=1e-4)

# compile model
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# reset states
reset_states = ResetStatesCallback()

#set up save best only checkpoint
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)

early_stopping = keras.callbacks.EarlyStopping(patience=50)

# fit model
model.fit(train_set, epochs=500,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint, reset_states])
```

    Epoch 1/500
        180/Unknown - 1s 8ms/step - loss: 1.7256e-04 - mae: 0.0138INFO:tensorflow:Assets written to: my_checkpoint/assets
    184/184 [==============================] - 9s 50ms/step - loss: 1.9735e-04 - mae: 0.0144 - val_loss: 0.0045 - val_mae: 0.0720
    Epoch 2/500
    178/184 [============================>.] - ETA: 0s - loss: 3.0633e-04 - mae: 0.0162INFO:tensorflow:Assets written to: my_checkpoint/assets
    184/184 [==============================] - 9s 49ms/step - loss: 3.0381e-04 - mae: 0.0162 - val_loss: 0.0020 - val_mae: 0.0469
    Epoch 3/500
    180/184 [============================>.] - ETA: 0s - loss: 1.2606e-04 - mae: 0.0117INFO:tensorflow:Assets written to: my_checkpoint/assets
    184/184 [==============================] - 9s 50ms/step - loss: 1.3149e-04 - mae: 0.0120 - val_loss: 0.0011 - val_mae: 0.0361
... truncated

    Epoch 192/500
    184/184 [==============================] - 2s 9ms/step - loss: 2.5988e-05 - mae: 0.0052 - val_loss: 1.6665e-04 - val_mae: 0.0139
    Epoch 193/500
    184/184 [==============================] - 2s 9ms/step - loss: 2.9839e-05 - mae: 0.0054 - val_loss: 1.6582e-04 - val_mae: 0.0143
    Epoch 194/500
    184/184 [==============================] - 2s 8ms/step - loss: 2.1207e-05 - mae: 0.0044 - val_loss: 1.0762e-04 - val_mae: 0.0111
    Epoch 195/500
    184/184 [==============================] - 2s 8ms/step - loss: 2.3044e-05 - mae: 0.0048 - val_loss: 1.4368e-04 - val_mae: 0.0139
    




    <tensorflow.python.keras.callbacks.History at 0x7fc122834208>



## Make Predictions


```python
# recall best model
model = keras.models.load_model("my_checkpoint")
```


```python
# make predictions
rnn_forecast = model.predict(normalized_x_test[np.newaxis,:])
rnn_forecast = rnn_forecast.flatten()

```


```python
# Example of how to iverse
rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()
rnn_unscaled_forecast.shape
```




    (422,)




```python
# plot results
plt.figure(figsize=(10,6))

plt.ylabel('Dollars $')
plt.xlabel('Timestep in Days')
plt.title(f'LSTM {window_size} Forecast')
plot_series(x_test.index, x_test, label="Actual")
plot_series(x_test.index, rnn_unscaled_forecast, label="Forecast")
plt.show()
```


![png](/assets/images/LSTM_Model/output_17_0.png)



```python
# calculate MAE
keras.metrics.mean_absolute_error(x_test, rnn_unscaled_forecast).numpy()
```




    1.1875452




```python

```
