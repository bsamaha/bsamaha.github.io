---
title: "Project: Part 3, Dense Forecast with Keras"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Keras
  - TensorFlow
---
**This notebook is part 3 of a multi-phase project of building a quantitative trading algorithm. Finally, this project is starting to get into the deep learning models.**

# Dense Forecasting

A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a. The following is te docstring of class Dense from the keras documentation:

output = activation(dot(input, kernel) + bias)where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.

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

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def window_dataset(series, window_size, batch_size=128,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
```

## Forecasting with Machine Learning

First, we will train a model to forecast the next step given the previous 30 steps, therefore, we need to create a dataset of 20-step windows for training. Every 20 steps is 1 trading month since the markets are only open Monday - Friday.


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
    


![png](/assets/images/Dense_Forecast/output_6_1.png)



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

### Dense Model Forecasting

#### Find Learning Rate


```python
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 20
train_set = window_dataset(normalized_x_train.flatten(), window_size)

model = keras.models.Sequential([
  keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  keras.layers.Dense(10, activation="relu"),
  keras.layers.Dense(1)
])

lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-7 * 10**(epoch / 20))
optimizer = keras.optimizers.Nadam(lr=1e-7)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

    Epoch 1/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2011 - mae: 0.5516
    Epoch 2/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.2011 - mae: 0.5515
    Epoch 3/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.2010 - mae: 0.5514
... truncated

    Epoch 97/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.4509e-04 - mae: 0.0122
    Epoch 98/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.5175e-04 - mae: 0.0124
    Epoch 99/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.4290e-04 - mae: 0.0126
    Epoch 100/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.7159e-04 - mae: 0.0131
    


```python
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-7, 1, 0, .1])
```




    (1e-07, 1.0, 0.0, 0.1)




![png](/assets/images/Dense_Forecast/output_11_1.png)


#### Create Model


```python
# Clear back end
keras.backend.clear_session()

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Set Window Size
window_size = 30
train_set = window_dataset(normalized_x_train.flatten(), window_size)
valid_set = window_dataset(normalized_x_valid.flatten(), window_size)

# Build 2 layer model with 10 neurons each and 1 output layer
model = keras.models.Sequential([
  keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  keras.layers.Dense(10, activation="relu"),
  keras.layers.Dense(1)
])

# Set optimizer
optimizer = keras.optimizers.Nadam(lr=1e-2)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Set early Stopping
early_stopping = keras.callbacks.EarlyStopping(patience=20)

# create save points for best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)

# Fit model
history = model.fit(train_set, epochs=500,
                    validation_data=valid_set,
                    callbacks=[early_stopping, model_checkpoint])
```

    Epoch 1/500
         36/Unknown - 1s 14ms/step - loss: 0.0015 - mae: 0.0411WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 34ms/step - loss: 0.0029 - mae: 0.0486 - val_loss: 6.9175e-04 - val_mae: 0.0291
    Epoch 2/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.4322e-04 - mae: 0.0164 - val_loss: 0.0030 - val_mae: 0.0714
    Epoch 3/500
    43/43 [==============================] - 1s 16ms/step - loss: 0.0020 - mae: 0.0329 - val_loss: 0.0034 - val_mae: 0.0752
    Epoch 4/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.1677e-04 - mae: 0.0156 - val_loss: 7.0642e-04 - val_mae: 0.0256
... truncated

    Epoch 124/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.0660e-04 - mae: 0.0104 - val_loss: 0.0044 - val_mae: 0.0912
    Epoch 125/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.8041e-05 - mae: 0.0099 - val_loss: 0.0026 - val_mae: 0.0693
    Epoch 126/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6590e-04 - mae: 0.0141 - val_loss: 0.0093 - val_mae: 0.1337
    


```python
model = keras.models.load_model("my_checkpoint")
```


```python
dense_forecast = model_forecast(model, spy_normalized_to_traindata.flatten()[x_test.index.min() - window_size:-1], window_size)[:, 0]
```


```python
spy_normalized_to_traindata.flatten().shape
```




    (6950,)




```python
# Undo the scaling
dense_forecast = x_train_scaler.inverse_transform(dense_forecast.reshape(-1,1)).flatten()
dense_forecast.shape
```




    (422,)




```python
# set style of charts
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = [10, 10]

plt.figure(figsize=(10, 6))
plt.title('Fully Dense Forecast')
plt.ylabel('Dollars $')
plt.xlabel('Timestep in Days')
plot_series(x_test.index, x_test)
plot_series(x_test.index, dense_forecast)
```


![png](/assets/images/Dense_Forecast/output_18_0.png)


#### Dense Model Result


```python
keras.metrics.mean_absolute_error(x_test, dense_forecast).numpy()
```




    5.4198823




```python

```
