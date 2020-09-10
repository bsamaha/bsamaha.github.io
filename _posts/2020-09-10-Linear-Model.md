---
title: "Project: Part 2, Time Series Forecasting with a Linear Model using Keras"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Keras
  - TensorFlow
---
**This notebook is part 2 of a multi-phase project of building a quantitative trading algorithm. This notebook explains what a linear model is using Keras.**

# This is Keras/TensorFlow right? I thought this was Deep Learning?

- Output = activation(dot(input, kernel) + bias

That looks familiar doesn't it? It looks almost identical to y = mx+b. The dot product is sum of the products in two sequences. Well, if there is only two sequences with a length of 1 then it is just the product of those two numbers. This simplifies down to the all to familiar y = mx + b.


## Setup


```python
# Example of dot product of two sequences of length 1
np.dot(2,1) + 5
```

    7




```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

keras = tf.keras

# set style of charts
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = [10, 10]
```


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
    


![png](/assets/images/Linear_Model_Forecast/output_7_1.png)

assets/images/Linear_Model_Forecast

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

### Linear Model


```python
# Clears any background saved info useful in notebooks
keras.backend.clear_session()

# Make reproducible 
tf.random.set_seed(42)
np.random.seed(42)

# set window size 
window_size = 20

# define training data (20 day windows shifted by 1 every time)
train_set = window_dataset(normalized_x_train.flatten(), window_size)

# Build Linear Model of a single dense layer
model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=[window_size])
])

# Find optimal learning rate
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(epoch / 30))
optimizer = keras.optimizers.Nadam(lr=1e-6)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Fit the model
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

    Epoch 1/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0482 - mae: 0.2671
    Epoch 2/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0481 - mae: 0.2668
    Epoch 3/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0480 - mae: 0.2664
... truncated

    Epoch 97/100
    43/43 [==============================] - 1s 12ms/step - loss: 2.5507e-04 - mae: 0.0166
    Epoch 98/100
    43/43 [==============================] - 1s 12ms/step - loss: 2.5321e-04 - mae: 0.0164
    Epoch 99/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.5107e-04 - mae: 0.0163
    Epoch 100/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.4980e-04 - mae: 0.0161
    


```python
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1, 0, .01])
```




    (1e-06, 1.0, 0.0, 0.01)




![png](/assets/images/Linear_Model_Forecast/output_11_1.png)



```python
# Useful to clear everything when rerunning cells
keras.backend.clear_session()

# Make this reproducible
tf.random.set_seed(42)
np.random.seed(42)

# Create train and validate windows
window_size = 20
train_set = window_dataset(normalized_x_train.flatten(), window_size)
valid_set = window_dataset(normalized_x_valid.flatten(), window_size)

# 1 layer producing linear output for 1 output from each window of 20 days
model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=[window_size]) # o
])

# Huber works well with "mae"
optimizer = keras.optimizers.Nadam(lr=1e-3)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# create save points for best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)

# Set up early stop
early_stopping = keras.callbacks.EarlyStopping(patience=10)

# fit model to data
model.fit(train_set, epochs=500,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint])


```

    Epoch 1/500
         41/Unknown - 0s 12ms/step - loss: 0.0980 - mae: 0.3945WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 25ms/step - loss: 0.0977 - mae: 0.3946 - val_loss: 0.3202 - val_mae: 0.7877
    Epoch 2/500
    36/43 [========================>.....] - ETA: 0s - loss: 0.0101 - mae: 0.1240INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 0.0108 - mae: 0.1285 - val_loss: 0.0474 - val_mae: 0.3004
    Epoch 3/500
    38/43 [=========================>....] - ETA: 0s - loss: 0.0011 - mae: 0.0384    INFO:tensorflow:Assets written to: 
    ... truncated
    Epoch 335/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5534e-05 - mae: 0.0059 - val_loss: 1.5882e-04 - val_mae: 0.0127
    Epoch 336/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.4622e-05 - mae: 0.0058 - val_loss: 1.3577e-04 - val_mae: 0.0114
    Epoch 337/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5224e-05 - mae: 0.0058 - val_loss: 1.4928e-04 - val_mae: 0.0121
    




    <tensorflow.python.keras.callbacks.History at 0x7f5281270978>




```python
lin_forecast = model_forecast(model, spy_normalized_to_traindata.flatten()[x_test.index.min() - window_size:-1], window_size)[:, 0]
```


```python
# Undo the scaling
lin_forecast = x_train_scaler.inverse_transform(lin_forecast.reshape(-1,1)).flatten()
lin_forecast.shape
```




    (422,)




```python
# Plot results
plt.title('Linear Forecast')
plt.ylabel('Dollars $')
plt.xlabel('Timestep in Days')
plot_series(x_test.index, x_test)
plot_series(x_test.index, lin_forecast)
```


![png](/assets/images/Linear_Model_Forecast/output_15_0.png)


#### Linear Model Result


```python
keras.metrics.mean_absolute_error(x_test, lin_forecast).numpy()
```




    3.8283994




```python

```
