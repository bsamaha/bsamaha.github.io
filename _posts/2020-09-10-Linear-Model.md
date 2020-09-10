---
title: "Project: Part 2, Time_Series_Forecasting_with_a_Linear_Model"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Keras
  - TensorFlow
---
**This notebook is part 1 of a multi-phase project of building a quantitative trading algorithm. This notebook starts with the more simple models which will be compared to more complex deep learning models. This notebook's last image shows the final results of all models tested.**

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
    Epoch 4/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0479 - mae: 0.2660
    Epoch 5/100
    43/43 [==============================] - 0s 12ms/step - loss: 0.0477 - mae: 0.2656
    Epoch 6/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0476 - mae: 0.2652
    Epoch 7/100
    43/43 [==============================] - 0s 12ms/step - loss: 0.0474 - mae: 0.2647
    Epoch 8/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0473 - mae: 0.2643
    Epoch 9/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0471 - mae: 0.2637
    Epoch 10/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0469 - mae: 0.2632
    Epoch 11/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0467 - mae: 0.2625
    Epoch 12/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0464 - mae: 0.2619
    Epoch 13/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0462 - mae: 0.2612
    Epoch 14/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0459 - mae: 0.2604
    Epoch 15/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0457 - mae: 0.2596
    Epoch 16/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0454 - mae: 0.2587
    Epoch 17/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0450 - mae: 0.2577
    Epoch 18/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0447 - mae: 0.2567
    Epoch 19/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0443 - mae: 0.2556
    Epoch 20/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0439 - mae: 0.2544
    Epoch 21/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0435 - mae: 0.2531
    Epoch 22/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0430 - mae: 0.2517
    Epoch 23/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0425 - mae: 0.2502
    Epoch 24/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0420 - mae: 0.2487
    Epoch 25/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0415 - mae: 0.2469
    Epoch 26/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0409 - mae: 0.2451
    Epoch 27/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0402 - mae: 0.2431
    Epoch 28/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0395 - mae: 0.2410
    Epoch 29/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0388 - mae: 0.2387
    Epoch 30/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0381 - mae: 0.2362
    Epoch 31/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0372 - mae: 0.2336
    Epoch 32/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0364 - mae: 0.2308
    Epoch 33/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0355 - mae: 0.2277
    Epoch 34/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0345 - mae: 0.2245
    Epoch 35/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0335 - mae: 0.2211
    Epoch 36/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0324 - mae: 0.2174
    Epoch 37/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0313 - mae: 0.2135
    Epoch 38/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0301 - mae: 0.2093
    Epoch 39/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0289 - mae: 0.2049
    Epoch 40/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0276 - mae: 0.2002
    Epoch 41/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0263 - mae: 0.1952
    Epoch 42/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0249 - mae: 0.1899
    Epoch 43/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0235 - mae: 0.1843
    Epoch 44/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0221 - mae: 0.1784
    Epoch 45/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0206 - mae: 0.1721
    Epoch 46/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0191 - mae: 0.1656
    Epoch 47/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0176 - mae: 0.1588
    Epoch 48/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0161 - mae: 0.1516
    Epoch 49/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0146 - mae: 0.1441
    Epoch 50/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0131 - mae: 0.1364
    Epoch 51/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0116 - mae: 0.1284
    Epoch 52/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0103 - mae: 0.1202
    Epoch 53/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0089 - mae: 0.1119
    Epoch 54/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0077 - mae: 0.1034
    Epoch 55/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0065 - mae: 0.0949
    Epoch 56/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0054 - mae: 0.0864
    Epoch 57/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0045 - mae: 0.0780
    Epoch 58/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0036 - mae: 0.0698
    Epoch 59/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0029 - mae: 0.0620
    Epoch 60/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0023 - mae: 0.0546
    Epoch 61/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0018 - mae: 0.0478
    Epoch 62/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0014 - mae: 0.0418
    Epoch 63/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0011 - mae: 0.0365
    Epoch 64/100
    43/43 [==============================] - 1s 13ms/step - loss: 8.2983e-04 - mae: 0.0322
    Epoch 65/100
    43/43 [==============================] - 1s 13ms/step - loss: 6.7028e-04 - mae: 0.0287
    Epoch 66/100
    43/43 [==============================] - 1s 13ms/step - loss: 5.6246e-04 - mae: 0.0262
    Epoch 67/100
    43/43 [==============================] - 1s 12ms/step - loss: 4.9305e-04 - mae: 0.0246
    Epoch 68/100
    43/43 [==============================] - 1s 12ms/step - loss: 4.5081e-04 - mae: 0.0236
    Epoch 69/100
    43/43 [==============================] - 1s 12ms/step - loss: 4.2603e-04 - mae: 0.0231
    Epoch 70/100
    43/43 [==============================] - 1s 12ms/step - loss: 4.1175e-04 - mae: 0.0229
    Epoch 71/100
    43/43 [==============================] - 1s 13ms/step - loss: 4.0334e-04 - mae: 0.0228
    Epoch 72/100
    43/43 [==============================] - 0s 11ms/step - loss: 3.9748e-04 - mae: 0.0227
    Epoch 73/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.9270e-04 - mae: 0.0226
    Epoch 74/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.8798e-04 - mae: 0.0225
    Epoch 75/100
    43/43 [==============================] - 1s 12ms/step - loss: 3.8325e-04 - mae: 0.0224
    Epoch 76/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.7776e-04 - mae: 0.0222
    Epoch 77/100
    43/43 [==============================] - 1s 12ms/step - loss: 3.7227e-04 - mae: 0.0221
    Epoch 78/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.6637e-04 - mae: 0.0219
    Epoch 79/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.6011e-04 - mae: 0.0216
    Epoch 80/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.5348e-04 - mae: 0.0214
    Epoch 81/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.4678e-04 - mae: 0.0212
    Epoch 82/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.3978e-04 - mae: 0.0209
    Epoch 83/100
    43/43 [==============================] - 1s 12ms/step - loss: 3.3256e-04 - mae: 0.0206
    Epoch 84/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.2544e-04 - mae: 0.0204
    Epoch 85/100
    43/43 [==============================] - 1s 14ms/step - loss: 3.1808e-04 - mae: 0.0201
    Epoch 86/100
    43/43 [==============================] - 1s 13ms/step - loss: 3.1101e-04 - mae: 0.0197
    Epoch 87/100
    43/43 [==============================] - 1s 12ms/step - loss: 3.0390e-04 - mae: 0.0194
    Epoch 88/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.9702e-04 - mae: 0.0191
    Epoch 89/100
    43/43 [==============================] - 1s 12ms/step - loss: 2.9046e-04 - mae: 0.0188
    Epoch 90/100
    43/43 [==============================] - 1s 15ms/step - loss: 2.8438e-04 - mae: 0.0185
    Epoch 91/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.7861e-04 - mae: 0.0182
    Epoch 92/100
    43/43 [==============================] - 0s 11ms/step - loss: 2.7321e-04 - mae: 0.0178
    Epoch 93/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.6875e-04 - mae: 0.0176
    Epoch 94/100
    43/43 [==============================] - 1s 12ms/step - loss: 2.6432e-04 - mae: 0.0173
    Epoch 95/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.6077e-04 - mae: 0.0171
    Epoch 96/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.5764e-04 - mae: 0.0168
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
    38/43 [=========================>....] - ETA: 0s - loss: 0.0011 - mae: 0.0384    INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 0.0011 - mae: 0.0395 - val_loss: 0.0097 - val_mae: 0.1326
    Epoch 4/500
    37/43 [========================>.....] - ETA: 0s - loss: 6.6360e-04 - mae: 0.0288INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 25ms/step - loss: 6.7882e-04 - mae: 0.0291 - val_loss: 0.0050 - val_mae: 0.0921
    Epoch 5/500
    43/43 [==============================] - ETA: 0s - loss: 6.5952e-04 - mae: 0.0292INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.5952e-04 - mae: 0.0292 - val_loss: 0.0040 - val_mae: 0.0817
    Epoch 6/500
    40/43 [==========================>...] - ETA: 0s - loss: 6.1500e-04 - mae: 0.0283INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.0881e-04 - mae: 0.0281 - val_loss: 0.0035 - val_mae: 0.0754
    Epoch 7/500
    38/43 [=========================>....] - ETA: 0s - loss: 5.6273e-04 - mae: 0.0271INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.5294e-04 - mae: 0.0268 - val_loss: 0.0030 - val_mae: 0.0697
    Epoch 8/500
    38/43 [=========================>....] - ETA: 0s - loss: 5.0974e-04 - mae: 0.0259INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.9784e-04 - mae: 0.0255 - val_loss: 0.0026 - val_mae: 0.0642
    Epoch 9/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.4726e-04 - mae: 0.0244INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.4503e-04 - mae: 0.0241 - val_loss: 0.0022 - val_mae: 0.0586
    Epoch 10/500
    39/43 [==========================>...] - ETA: 0s - loss: 4.0477e-04 - mae: 0.0231INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 3.9675e-04 - mae: 0.0228 - val_loss: 0.0019 - val_mae: 0.0537
    Epoch 11/500
    43/43 [==============================] - ETA: 0s - loss: 3.5191e-04 - mae: 0.0215INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 3.5191e-04 - mae: 0.0215 - val_loss: 0.0016 - val_mae: 0.0490
    Epoch 12/500
    36/43 [========================>.....] - ETA: 0s - loss: 3.1567e-04 - mae: 0.0206INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 3.1136e-04 - mae: 0.0202 - val_loss: 0.0014 - val_mae: 0.0443
    Epoch 13/500
    37/43 [========================>.....] - ETA: 0s - loss: 2.8023e-04 - mae: 0.0194INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 2.7667e-04 - mae: 0.0191 - val_loss: 0.0012 - val_mae: 0.0406
    Epoch 14/500
    40/43 [==========================>...] - ETA: 0s - loss: 2.4411e-04 - mae: 0.0181INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 2.4489e-04 - mae: 0.0180 - val_loss: 0.0010 - val_mae: 0.0371
    Epoch 15/500
    43/43 [==============================] - ETA: 0s - loss: 2.1743e-04 - mae: 0.0169INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 2.1743e-04 - mae: 0.0169 - val_loss: 8.8185e-04 - val_mae: 0.0335
    Epoch 16/500
    42/43 [============================>.] - ETA: 0s - loss: 1.9628e-04 - mae: 0.0161INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 1.9545e-04 - mae: 0.0160 - val_loss: 7.8419e-04 - val_mae: 0.0310
    Epoch 17/500
    43/43 [==============================] - ETA: 0s - loss: 1.7493e-04 - mae: 0.0152INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 1.7493e-04 - mae: 0.0152 - val_loss: 6.9850e-04 - val_mae: 0.0287
    Epoch 18/500
    37/43 [========================>.....] - ETA: 0s - loss: 1.5659e-04 - mae: 0.0144INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.5889e-04 - mae: 0.0143 - val_loss: 6.2953e-04 - val_mae: 0.0268
    Epoch 19/500
    41/43 [===========================>..] - ETA: 0s - loss: 1.4449e-04 - mae: 0.0137INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.4489e-04 - mae: 0.0136 - val_loss: 5.6843e-04 - val_mae: 0.0251
    Epoch 20/500
    38/43 [=========================>....] - ETA: 0s - loss: 1.3485e-04 - mae: 0.0131INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.3455e-04 - mae: 0.0130 - val_loss: 5.3348e-04 - val_mae: 0.0240
    Epoch 21/500
    38/43 [=========================>....] - ETA: 0s - loss: 1.2077e-04 - mae: 0.0123INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.2518e-04 - mae: 0.0125 - val_loss: 5.0010e-04 - val_mae: 0.0230
    Epoch 22/500
    39/43 [==========================>...] - ETA: 0s - loss: 1.1480e-04 - mae: 0.0119INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.1782e-04 - mae: 0.0120 - val_loss: 4.6347e-04 - val_mae: 0.0218
    Epoch 23/500
    42/43 [============================>.] - ETA: 0s - loss: 1.1294e-04 - mae: 0.0116INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 25ms/step - loss: 1.1295e-04 - mae: 0.0116 - val_loss: 4.5285e-04 - val_mae: 0.0215
    Epoch 24/500
    39/43 [==========================>...] - ETA: 0s - loss: 1.0346e-04 - mae: 0.0110INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.0773e-04 - mae: 0.0112 - val_loss: 4.3113e-04 - val_mae: 0.0208
    Epoch 25/500
    38/43 [=========================>....] - ETA: 0s - loss: 1.0082e-04 - mae: 0.0107INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 1.0478e-04 - mae: 0.0109 - val_loss: 4.2231e-04 - val_mae: 0.0205
    Epoch 26/500
    43/43 [==============================] - 1s 16ms/step - loss: 1.0204e-04 - mae: 0.0106 - val_loss: 4.2283e-04 - val_mae: 0.0206
    Epoch 27/500
    37/43 [========================>.....] - ETA: 0s - loss: 9.2812e-05 - mae: 0.0100INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.9561e-05 - mae: 0.0104 - val_loss: 4.0803e-04 - val_mae: 0.0201
    Epoch 28/500
    41/43 [===========================>..] - ETA: 0s - loss: 9.6658e-05 - mae: 0.0101INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.8380e-05 - mae: 0.0102 - val_loss: 4.0414e-04 - val_mae: 0.0200
    Epoch 29/500
    42/43 [============================>.] - ETA: 0s - loss: 9.6481e-05 - mae: 0.0100INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.7127e-05 - mae: 0.0101 - val_loss: 3.9956e-04 - val_mae: 0.0199
    Epoch 30/500
    36/43 [========================>.....] - ETA: 0s - loss: 9.0746e-05 - mae: 0.0096INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.6381e-05 - mae: 0.0100 - val_loss: 3.9822e-04 - val_mae: 0.0198
    Epoch 31/500
    37/43 [========================>.....] - ETA: 0s - loss: 9.0412e-05 - mae: 0.0095INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.5616e-05 - mae: 0.0099 - val_loss: 3.9581e-04 - val_mae: 0.0198
    Epoch 32/500
    40/43 [==========================>...] - ETA: 0s - loss: 9.1131e-05 - mae: 0.0096INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.5084e-05 - mae: 0.0098 - val_loss: 3.9402e-04 - val_mae: 0.0197
    Epoch 33/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.9228e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 26ms/step - loss: 9.4662e-05 - mae: 0.0097 - val_loss: 3.9190e-04 - val_mae: 0.0197
    Epoch 34/500
    39/43 [==========================>...] - ETA: 0s - loss: 9.1858e-05 - mae: 0.0095INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 9.4527e-05 - mae: 0.0097 - val_loss: 3.9167e-04 - val_mae: 0.0197
    Epoch 35/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.4189e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.4178e-05 - mae: 0.0097 - val_loss: 3.9095e-04 - val_mae: 0.0197
    Epoch 36/500
    40/43 [==========================>...] - ETA: 0s - loss: 9.1108e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.4026e-05 - mae: 0.0096 - val_loss: 3.9048e-04 - val_mae: 0.0197
    Epoch 37/500
    43/43 [==============================] - 1s 15ms/step - loss: 9.3857e-05 - mae: 0.0096 - val_loss: 3.9189e-04 - val_mae: 0.0197
    Epoch 38/500
    41/43 [===========================>..] - ETA: 0s - loss: 9.2022e-05 - mae: 0.0095INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.3635e-05 - mae: 0.0096 - val_loss: 3.8965e-04 - val_mae: 0.0197
    Epoch 39/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.3609e-05 - mae: 0.0096 - val_loss: 3.8967e-04 - val_mae: 0.0196
    Epoch 40/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.5776e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.3430e-05 - mae: 0.0096 - val_loss: 3.8951e-04 - val_mae: 0.0196
    Epoch 41/500
    35/43 [=======================>......] - ETA: 0s - loss: 8.5766e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.3308e-05 - mae: 0.0096 - val_loss: 3.8845e-04 - val_mae: 0.0196
    Epoch 42/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.3200e-05 - mae: 0.0096 - val_loss: 3.8865e-04 - val_mae: 0.0196
    Epoch 43/500
    39/43 [==========================>...] - ETA: 0s - loss: 8.9380e-05 - mae: 0.0093INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.3113e-05 - mae: 0.0096 - val_loss: 3.8748e-04 - val_mae: 0.0196
    Epoch 44/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.3002e-05 - mae: 0.0095 - val_loss: 3.8749e-04 - val_mae: 0.0197
    Epoch 45/500
    40/43 [==========================>...] - ETA: 0s - loss: 9.0179e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.2881e-05 - mae: 0.0095 - val_loss: 3.8664e-04 - val_mae: 0.0196
    Epoch 46/500
    40/43 [==========================>...] - ETA: 0s - loss: 9.0629e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.2753e-05 - mae: 0.0095 - val_loss: 3.8597e-04 - val_mae: 0.0196
    Epoch 47/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.2672e-05 - mae: 0.0095 - val_loss: 3.8606e-04 - val_mae: 0.0195
    Epoch 48/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.7467e-05 - mae: 0.0092INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.2540e-05 - mae: 0.0095 - val_loss: 3.8543e-04 - val_mae: 0.0196
    Epoch 49/500
    43/43 [==============================] - ETA: 0s - loss: 9.2482e-05 - mae: 0.0095INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.2482e-05 - mae: 0.0095 - val_loss: 3.8443e-04 - val_mae: 0.0195
    Epoch 50/500
    43/43 [==============================] - ETA: 0s - loss: 9.2285e-05 - mae: 0.0095INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.2285e-05 - mae: 0.0095 - val_loss: 3.8394e-04 - val_mae: 0.0195
    Epoch 51/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.2139e-05 - mae: 0.0095 - val_loss: 3.8423e-04 - val_mae: 0.0196
    Epoch 52/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.2029e-05 - mae: 0.0095 - val_loss: 3.8559e-04 - val_mae: 0.0195
    Epoch 53/500
    40/43 [==========================>...] - ETA: 0s - loss: 9.0543e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.2008e-05 - mae: 0.0095 - val_loss: 3.8254e-04 - val_mae: 0.0195
    Epoch 54/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.5256e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.1773e-05 - mae: 0.0095 - val_loss: 3.8166e-04 - val_mae: 0.0195
    Epoch 55/500
    40/43 [==========================>...] - ETA: 0s - loss: 9.0364e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.1602e-05 - mae: 0.0095 - val_loss: 3.8120e-04 - val_mae: 0.0194
    Epoch 56/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.7426e-05 - mae: 0.0092INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.1489e-05 - mae: 0.0095 - val_loss: 3.8095e-04 - val_mae: 0.0194
    Epoch 57/500
    43/43 [==============================] - 1s 15ms/step - loss: 9.1295e-05 - mae: 0.0095 - val_loss: 3.8148e-04 - val_mae: 0.0194
    Epoch 58/500
    42/43 [============================>.] - ETA: 0s - loss: 9.0805e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.1172e-05 - mae: 0.0095 - val_loss: 3.7982e-04 - val_mae: 0.0195
    Epoch 59/500
    39/43 [==========================>...] - ETA: 0s - loss: 8.7581e-05 - mae: 0.0092INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.1163e-05 - mae: 0.0094 - val_loss: 3.7851e-04 - val_mae: 0.0194
    Epoch 60/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.0920e-05 - mae: 0.0094 - val_loss: 3.8002e-04 - val_mae: 0.0194
    Epoch 61/500
    42/43 [============================>.] - ETA: 0s - loss: 8.9704e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.0782e-05 - mae: 0.0094 - val_loss: 3.7833e-04 - val_mae: 0.0193
    Epoch 62/500
    43/43 [==============================] - ETA: 0s - loss: 9.0561e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.0561e-05 - mae: 0.0094 - val_loss: 3.7676e-04 - val_mae: 0.0193
    Epoch 63/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.3449e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 9.0465e-05 - mae: 0.0094 - val_loss: 3.7668e-04 - val_mae: 0.0193
    Epoch 64/500
    40/43 [==========================>...] - ETA: 0s - loss: 8.7583e-05 - mae: 0.0092INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 9.0306e-05 - mae: 0.0094 - val_loss: 3.7515e-04 - val_mae: 0.0193
    Epoch 65/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.0084e-05 - mae: 0.0094 - val_loss: 3.7544e-04 - val_mae: 0.0194
    Epoch 66/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.3413e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 9.0045e-05 - mae: 0.0094 - val_loss: 3.7438e-04 - val_mae: 0.0193
    Epoch 67/500
    42/43 [============================>.] - ETA: 0s - loss: 8.8880e-05 - mae: 0.0093INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.9814e-05 - mae: 0.0094 - val_loss: 3.7314e-04 - val_mae: 0.0193
    Epoch 68/500
    39/43 [==========================>...] - ETA: 0s - loss: 8.5442e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 8.9670e-05 - mae: 0.0094 - val_loss: 3.7222e-04 - val_mae: 0.0192
    Epoch 69/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.5386e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.9465e-05 - mae: 0.0094 - val_loss: 3.7145e-04 - val_mae: 0.0192
    Epoch 70/500
    43/43 [==============================] - ETA: 0s - loss: 8.9263e-05 - mae: 0.0094INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.9263e-05 - mae: 0.0094 - val_loss: 3.7077e-04 - val_mae: 0.0192
    Epoch 71/500
    39/43 [==========================>...] - ETA: 0s - loss: 8.5926e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 8.9051e-05 - mae: 0.0093 - val_loss: 3.7026e-04 - val_mae: 0.0192
    Epoch 72/500
    39/43 [==========================>...] - ETA: 0s - loss: 8.6600e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.8899e-05 - mae: 0.0093 - val_loss: 3.6908e-04 - val_mae: 0.0191
    Epoch 73/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.3376e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.8775e-05 - mae: 0.0093 - val_loss: 3.6827e-04 - val_mae: 0.0191
    Epoch 74/500
    37/43 [========================>.....] - ETA: 0s - loss: 8.5477e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.8534e-05 - mae: 0.0093 - val_loss: 3.6750e-04 - val_mae: 0.0191
    Epoch 75/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.3429e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.8279e-05 - mae: 0.0093 - val_loss: 3.6694e-04 - val_mae: 0.0191
    Epoch 76/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.4927e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 8.8143e-05 - mae: 0.0093 - val_loss: 3.6575e-04 - val_mae: 0.0191
    Epoch 77/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.1961e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.7869e-05 - mae: 0.0093 - val_loss: 3.6558e-04 - val_mae: 0.0191
    Epoch 78/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.7777e-05 - mae: 0.0093 - val_loss: 3.6715e-04 - val_mae: 0.0192
    Epoch 79/500
    38/43 [=========================>....] - ETA: 0s - loss: 8.2550e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 8.7625e-05 - mae: 0.0093 - val_loss: 3.6314e-04 - val_mae: 0.0190
    Epoch 80/500
    41/43 [===========================>..] - ETA: 0s - loss: 8.5537e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 8.7342e-05 - mae: 0.0093 - val_loss: 3.6221e-04 - val_mae: 0.0190
    Epoch 81/500
    43/43 [==============================] - 1s 15ms/step - loss: 8.7076e-05 - mae: 0.0092 - val_loss: 3.6409e-04 - val_mae: 0.0190
    Epoch 82/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.1851e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.6960e-05 - mae: 0.0092 - val_loss: 3.6034e-04 - val_mae: 0.0189
    Epoch 83/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.6747e-05 - mae: 0.0092 - val_loss: 3.6176e-04 - val_mae: 0.0190
    Epoch 84/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.6388e-05 - mae: 0.0092 - val_loss: 3.6298e-04 - val_mae: 0.0191
    Epoch 85/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.6294e-05 - mae: 0.0092 - val_loss: 3.6152e-04 - val_mae: 0.0191
    Epoch 86/500
    36/43 [========================>.....] - ETA: 0s - loss: 8.0329e-05 - mae: 0.0088INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.6096e-05 - mae: 0.0092 - val_loss: 3.5655e-04 - val_mae: 0.0188
    Epoch 87/500
    40/43 [==========================>...] - ETA: 0s - loss: 8.3727e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.5895e-05 - mae: 0.0092 - val_loss: 3.5573e-04 - val_mae: 0.0188
    Epoch 88/500
    43/43 [==============================] - ETA: 0s - loss: 8.5606e-05 - mae: 0.0092INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.5606e-05 - mae: 0.0092 - val_loss: 3.5469e-04 - val_mae: 0.0188
    Epoch 89/500
    39/43 [==========================>...] - ETA: 0s - loss: 8.1223e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.5349e-05 - mae: 0.0092 - val_loss: 3.5431e-04 - val_mae: 0.0187
    Epoch 90/500
    41/43 [===========================>..] - ETA: 0s - loss: 8.4190e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.5227e-05 - mae: 0.0091 - val_loss: 3.5408e-04 - val_mae: 0.0187
    Epoch 91/500
    43/43 [==============================] - ETA: 0s - loss: 8.5113e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.5113e-05 - mae: 0.0091 - val_loss: 3.5194e-04 - val_mae: 0.0187
    Epoch 92/500
    42/43 [============================>.] - ETA: 0s - loss: 8.4030e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.4657e-05 - mae: 0.0091 - val_loss: 3.5125e-04 - val_mae: 0.0187
    Epoch 93/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.4404e-05 - mae: 0.0091 - val_loss: 3.5192e-04 - val_mae: 0.0187
    Epoch 94/500
    41/43 [===========================>..] - ETA: 0s - loss: 8.3087e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 26ms/step - loss: 8.4431e-05 - mae: 0.0091 - val_loss: 3.4819e-04 - val_mae: 0.0186
    Epoch 95/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.3890e-05 - mae: 0.0091 - val_loss: 3.5367e-04 - val_mae: 0.0189
    Epoch 96/500
    42/43 [============================>.] - ETA: 0s - loss: 8.3984e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.3713e-05 - mae: 0.0091 - val_loss: 3.4606e-04 - val_mae: 0.0185
    Epoch 97/500
    43/43 [==============================] - 1s 15ms/step - loss: 8.3382e-05 - mae: 0.0091 - val_loss: 3.4761e-04 - val_mae: 0.0186
    Epoch 98/500
    38/43 [=========================>....] - ETA: 0s - loss: 7.7595e-05 - mae: 0.0087INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.3230e-05 - mae: 0.0090 - val_loss: 3.4386e-04 - val_mae: 0.0184
    Epoch 99/500
    42/43 [============================>.] - ETA: 0s - loss: 8.2816e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.2821e-05 - mae: 0.0090 - val_loss: 3.4266e-04 - val_mae: 0.0184
    Epoch 100/500
    43/43 [==============================] - 1s 15ms/step - loss: 8.2555e-05 - mae: 0.0090 - val_loss: 3.4298e-04 - val_mae: 0.0184
    Epoch 101/500
    39/43 [==========================>...] - ETA: 0s - loss: 7.9975e-05 - mae: 0.0088INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.2365e-05 - mae: 0.0090 - val_loss: 3.4043e-04 - val_mae: 0.0183
    Epoch 102/500
    36/43 [========================>.....] - ETA: 0s - loss: 7.6072e-05 - mae: 0.0085INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.2103e-05 - mae: 0.0090 - val_loss: 3.3911e-04 - val_mae: 0.0183
    Epoch 103/500
    38/43 [=========================>....] - ETA: 0s - loss: 7.7725e-05 - mae: 0.0087INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.1865e-05 - mae: 0.0090 - val_loss: 3.3838e-04 - val_mae: 0.0183
    Epoch 104/500
    43/43 [==============================] - ETA: 0s - loss: 8.1661e-05 - mae: 0.0090INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.1661e-05 - mae: 0.0090 - val_loss: 3.3671e-04 - val_mae: 0.0182
    Epoch 105/500
    43/43 [==============================] - 1s 15ms/step - loss: 8.1339e-05 - mae: 0.0089 - val_loss: 3.3864e-04 - val_mae: 0.0184
    Epoch 106/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.9305e-05 - mae: 0.0088INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.1028e-05 - mae: 0.0089 - val_loss: 3.3635e-04 - val_mae: 0.0183
    Epoch 107/500
    38/43 [=========================>....] - ETA: 0s - loss: 7.6502e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.0825e-05 - mae: 0.0089 - val_loss: 3.3339e-04 - val_mae: 0.0181
    Epoch 108/500
    36/43 [========================>.....] - ETA: 0s - loss: 7.3979e-05 - mae: 0.0084INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 8.0423e-05 - mae: 0.0089 - val_loss: 3.3280e-04 - val_mae: 0.0181
    Epoch 109/500
    38/43 [=========================>....] - ETA: 0s - loss: 7.7069e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 8.0279e-05 - mae: 0.0089 - val_loss: 3.3164e-04 - val_mae: 0.0181
    Epoch 110/500
    43/43 [==============================] - ETA: 0s - loss: 7.9809e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.9809e-05 - mae: 0.0089 - val_loss: 3.2952e-04 - val_mae: 0.0181
    Epoch 111/500
    39/43 [==========================>...] - ETA: 0s - loss: 7.6811e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 24ms/step - loss: 7.9611e-05 - mae: 0.0088 - val_loss: 3.2801e-04 - val_mae: 0.0180
    Epoch 112/500
    43/43 [==============================] - ETA: 0s - loss: 7.9311e-05 - mae: 0.0088INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.9311e-05 - mae: 0.0088 - val_loss: 3.2753e-04 - val_mae: 0.0180
    Epoch 113/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.9010e-05 - mae: 0.0088 - val_loss: 3.2913e-04 - val_mae: 0.0181
    Epoch 114/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.8828e-05 - mae: 0.0088 - val_loss: 3.2773e-04 - val_mae: 0.0181
    Epoch 115/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.8595e-05 - mae: 0.0088 - val_loss: 3.2876e-04 - val_mae: 0.0181
    Epoch 116/500
    40/43 [==========================>...] - ETA: 0s - loss: 7.6205e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 7.8268e-05 - mae: 0.0088 - val_loss: 3.2445e-04 - val_mae: 0.0180
    Epoch 117/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.7895e-05 - mae: 0.0088 - val_loss: 3.2644e-04 - val_mae: 0.0181
    Epoch 118/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.7066e-05 - mae: 0.0087INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 7.7762e-05 - mae: 0.0087 - val_loss: 3.2083e-04 - val_mae: 0.0178
    Epoch 119/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.7540e-05 - mae: 0.0087 - val_loss: 3.4483e-04 - val_mae: 0.0189
    Epoch 120/500
    39/43 [==========================>...] - ETA: 0s - loss: 7.4769e-05 - mae: 0.0085INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 24ms/step - loss: 7.7466e-05 - mae: 0.0087 - val_loss: 3.1622e-04 - val_mae: 0.0177
    Epoch 121/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.6675e-05 - mae: 0.0087 - val_loss: 3.1946e-04 - val_mae: 0.0178
    Epoch 122/500
    39/43 [==========================>...] - ETA: 0s - loss: 7.4435e-05 - mae: 0.0085INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 25ms/step - loss: 7.6775e-05 - mae: 0.0087 - val_loss: 3.1496e-04 - val_mae: 0.0177
    Epoch 123/500
    37/43 [========================>.....] - ETA: 0s - loss: 6.9928e-05 - mae: 0.0082INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.6076e-05 - mae: 0.0087 - val_loss: 3.1343e-04 - val_mae: 0.0176
    Epoch 124/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.4338e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.5927e-05 - mae: 0.0087 - val_loss: 3.1096e-04 - val_mae: 0.0175
    Epoch 125/500
    43/43 [==============================] - ETA: 0s - loss: 7.5660e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 7.5660e-05 - mae: 0.0086 - val_loss: 3.1030e-04 - val_mae: 0.0175
    Epoch 126/500
    43/43 [==============================] - 1s 15ms/step - loss: 7.5365e-05 - mae: 0.0086 - val_loss: 3.1069e-04 - val_mae: 0.0175
    Epoch 127/500
    36/43 [========================>.....] - ETA: 0s - loss: 6.8435e-05 - mae: 0.0081INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.4861e-05 - mae: 0.0086 - val_loss: 3.1030e-04 - val_mae: 0.0176
    Epoch 128/500
    43/43 [==============================] - ETA: 0s - loss: 7.4566e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.4566e-05 - mae: 0.0086 - val_loss: 3.0837e-04 - val_mae: 0.0175
    Epoch 129/500
    38/43 [=========================>....] - ETA: 0s - loss: 7.0294e-05 - mae: 0.0083INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.4564e-05 - mae: 0.0086 - val_loss: 3.0403e-04 - val_mae: 0.0173
    Epoch 130/500
    39/43 [==========================>...] - ETA: 0s - loss: 7.1271e-05 - mae: 0.0083INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.3965e-05 - mae: 0.0085 - val_loss: 3.0320e-04 - val_mae: 0.0173
    Epoch 131/500
    43/43 [==============================] - 1s 15ms/step - loss: 7.3568e-05 - mae: 0.0085 - val_loss: 3.1054e-04 - val_mae: 0.0177
    Epoch 132/500
    42/43 [============================>.] - ETA: 0s - loss: 7.3592e-05 - mae: 0.0085INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 7.3402e-05 - mae: 0.0085 - val_loss: 3.0069e-04 - val_mae: 0.0172
    Epoch 133/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.3162e-05 - mae: 0.0085 - val_loss: 3.0446e-04 - val_mae: 0.0174
    Epoch 134/500
    42/43 [============================>.] - ETA: 0s - loss: 7.2309e-05 - mae: 0.0084INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 7.2770e-05 - mae: 0.0085 - val_loss: 2.9738e-04 - val_mae: 0.0172
    Epoch 135/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.0795e-05 - mae: 0.0083INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 7.2600e-05 - mae: 0.0085 - val_loss: 2.9526e-04 - val_mae: 0.0170
    Epoch 136/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.2013e-05 - mae: 0.0084 - val_loss: 2.9744e-04 - val_mae: 0.0172
    Epoch 137/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.1774e-05 - mae: 0.0084 - val_loss: 2.9763e-04 - val_mae: 0.0172
    Epoch 138/500
    39/43 [==========================>...] - ETA: 0s - loss: 6.7922e-05 - mae: 0.0082INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 7.1660e-05 - mae: 0.0084 - val_loss: 2.9088e-04 - val_mae: 0.0169
    Epoch 139/500
    42/43 [============================>.] - ETA: 0s - loss: 7.0485e-05 - mae: 0.0083INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.1005e-05 - mae: 0.0084 - val_loss: 2.8939e-04 - val_mae: 0.0169
    Epoch 140/500
    38/43 [=========================>....] - ETA: 0s - loss: 6.7687e-05 - mae: 0.0081INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 7.0933e-05 - mae: 0.0084 - val_loss: 2.8883e-04 - val_mae: 0.0169
    Epoch 141/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.0527e-05 - mae: 0.0083 - val_loss: 2.9000e-04 - val_mae: 0.0169
    Epoch 142/500
    40/43 [==========================>...] - ETA: 0s - loss: 6.8368e-05 - mae: 0.0081INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 7.0383e-05 - mae: 0.0083 - val_loss: 2.8555e-04 - val_mae: 0.0168
    Epoch 143/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.9950e-05 - mae: 0.0083 - val_loss: 2.9368e-04 - val_mae: 0.0171
    Epoch 144/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.9798e-05 - mae: 0.0083 - val_loss: 2.8724e-04 - val_mae: 0.0170
    Epoch 145/500
    37/43 [========================>.....] - ETA: 0s - loss: 6.4706e-05 - mae: 0.0079INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.9399e-05 - mae: 0.0083 - val_loss: 2.8326e-04 - val_mae: 0.0167
    Epoch 146/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.8882e-05 - mae: 0.0082INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.9608e-05 - mae: 0.0083 - val_loss: 2.7929e-04 - val_mae: 0.0166
    Epoch 147/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.8853e-05 - mae: 0.0082 - val_loss: 2.9156e-04 - val_mae: 0.0171
    Epoch 148/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.8231e-05 - mae: 0.0082 - val_loss: 2.8302e-04 - val_mae: 0.0167
    Epoch 149/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.8151e-05 - mae: 0.0082 - val_loss: 2.8341e-04 - val_mae: 0.0167
    Epoch 150/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.7882e-05 - mae: 0.0082 - val_loss: 2.9096e-04 - val_mae: 0.0174
    Epoch 151/500
    43/43 [==============================] - ETA: 0s - loss: 6.8475e-05 - mae: 0.0082INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.8475e-05 - mae: 0.0082 - val_loss: 2.7216e-04 - val_mae: 0.0163
    Epoch 152/500
    40/43 [==========================>...] - ETA: 0s - loss: 6.5778e-05 - mae: 0.0080INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.7229e-05 - mae: 0.0081 - val_loss: 2.7069e-04 - val_mae: 0.0163
    Epoch 153/500
    35/43 [=======================>......] - ETA: 0s - loss: 6.0627e-05 - mae: 0.0076INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 6.6961e-05 - mae: 0.0081 - val_loss: 2.7025e-04 - val_mae: 0.0164
    Epoch 154/500
    37/43 [========================>.....] - ETA: 0s - loss: 6.4650e-05 - mae: 0.0079INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.6499e-05 - mae: 0.0081 - val_loss: 2.6776e-04 - val_mae: 0.0162
    Epoch 155/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.6518e-05 - mae: 0.0081 - val_loss: 2.7196e-04 - val_mae: 0.0166
    Epoch 156/500
    36/43 [========================>.....] - ETA: 0s - loss: 6.1882e-05 - mae: 0.0077INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.6656e-05 - mae: 0.0081 - val_loss: 2.6533e-04 - val_mae: 0.0161
    Epoch 157/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.5654e-05 - mae: 0.0080 - val_loss: 2.6588e-04 - val_mae: 0.0161
    Epoch 158/500
    38/43 [=========================>....] - ETA: 0s - loss: 6.1697e-05 - mae: 0.0077INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.5304e-05 - mae: 0.0080 - val_loss: 2.6519e-04 - val_mae: 0.0161
    Epoch 159/500
    43/43 [==============================] - ETA: 0s - loss: 6.5151e-05 - mae: 0.0080INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.5151e-05 - mae: 0.0080 - val_loss: 2.6067e-04 - val_mae: 0.0160
    Epoch 160/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.4971e-05 - mae: 0.0080 - val_loss: 2.8013e-04 - val_mae: 0.0172
    Epoch 161/500
    40/43 [==========================>...] - ETA: 0s - loss: 6.3228e-05 - mae: 0.0079INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.4746e-05 - mae: 0.0080 - val_loss: 2.5876e-04 - val_mae: 0.0160
    Epoch 162/500
    38/43 [=========================>....] - ETA: 0s - loss: 6.1516e-05 - mae: 0.0077INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 6.4187e-05 - mae: 0.0080 - val_loss: 2.5756e-04 - val_mae: 0.0159
    Epoch 163/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.3805e-05 - mae: 0.0079 - val_loss: 2.5912e-04 - val_mae: 0.0159
    Epoch 164/500
    38/43 [=========================>....] - ETA: 0s - loss: 6.0229e-05 - mae: 0.0076INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.3858e-05 - mae: 0.0079 - val_loss: 2.5408e-04 - val_mae: 0.0158
    Epoch 165/500
    36/43 [========================>.....] - ETA: 0s - loss: 5.7819e-05 - mae: 0.0074INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.3178e-05 - mae: 0.0079 - val_loss: 2.5338e-04 - val_mae: 0.0159
    Epoch 166/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.2732e-05 - mae: 0.0079INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 6.3623e-05 - mae: 0.0079 - val_loss: 2.5103e-04 - val_mae: 0.0157
    Epoch 167/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.2931e-05 - mae: 0.0079 - val_loss: 2.5884e-04 - val_mae: 0.0160
    Epoch 168/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.2587e-05 - mae: 0.0079 - val_loss: 2.5796e-04 - val_mae: 0.0159
    Epoch 169/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.2146e-05 - mae: 0.0078 - val_loss: 2.6965e-04 - val_mae: 0.0164
    Epoch 170/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.2031e-05 - mae: 0.0078 - val_loss: 2.7356e-04 - val_mae: 0.0166
    Epoch 171/500
    38/43 [=========================>....] - ETA: 0s - loss: 5.8092e-05 - mae: 0.0075INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.1573e-05 - mae: 0.0078 - val_loss: 2.4710e-04 - val_mae: 0.0158
    Epoch 172/500
    40/43 [==========================>...] - ETA: 0s - loss: 5.8818e-05 - mae: 0.0076INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 24ms/step - loss: 6.0955e-05 - mae: 0.0077 - val_loss: 2.4423e-04 - val_mae: 0.0154
    Epoch 173/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.2539e-05 - mae: 0.0078 - val_loss: 2.6795e-04 - val_mae: 0.0170
    Epoch 174/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.1155e-05 - mae: 0.0078 - val_loss: 2.4625e-04 - val_mae: 0.0158
    Epoch 175/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.1881e-05 - mae: 0.0078INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 25ms/step - loss: 6.2858e-05 - mae: 0.0079 - val_loss: 2.4243e-04 - val_mae: 0.0154
    Epoch 176/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.0344e-05 - mae: 0.0077 - val_loss: 2.4676e-04 - val_mae: 0.0159
    Epoch 177/500
    36/43 [========================>.....] - ETA: 0s - loss: 5.3977e-05 - mae: 0.0072INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 6.0162e-05 - mae: 0.0077 - val_loss: 2.3919e-04 - val_mae: 0.0153
    Epoch 178/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.3011e-05 - mae: 0.0079 - val_loss: 2.9291e-04 - val_mae: 0.0185
    Epoch 179/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.9457e-05 - mae: 0.0077 - val_loss: 2.4332e-04 - val_mae: 0.0159
    Epoch 180/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.8875e-05 - mae: 0.0076 - val_loss: 2.5237e-04 - val_mae: 0.0158
    Epoch 181/500
    43/43 [==============================] - ETA: 0s - loss: 5.9029e-05 - mae: 0.0076INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 5.9029e-05 - mae: 0.0076 - val_loss: 2.3471e-04 - val_mae: 0.0154
    Epoch 182/500
    38/43 [=========================>....] - ETA: 0s - loss: 5.5980e-05 - mae: 0.0074INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.8184e-05 - mae: 0.0076 - val_loss: 2.3322e-04 - val_mae: 0.0151
    Epoch 183/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.0485e-05 - mae: 0.0077 - val_loss: 2.9808e-04 - val_mae: 0.0189
    Epoch 184/500
    36/43 [========================>.....] - ETA: 0s - loss: 5.3600e-05 - mae: 0.0072INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.8155e-05 - mae: 0.0076 - val_loss: 2.2838e-04 - val_mae: 0.0149
    Epoch 185/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.9666e-05 - mae: 0.0077 - val_loss: 2.4330e-04 - val_mae: 0.0155
    Epoch 186/500
    41/43 [===========================>..] - ETA: 0s - loss: 5.6623e-05 - mae: 0.0074INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 5.7385e-05 - mae: 0.0075 - val_loss: 2.2557e-04 - val_mae: 0.0149
    Epoch 187/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.8340e-05 - mae: 0.0076 - val_loss: 2.6844e-04 - val_mae: 0.0175
    Epoch 188/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.7353e-05 - mae: 0.0075 - val_loss: 2.3129e-04 - val_mae: 0.0154
    Epoch 189/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.6811e-05 - mae: 0.0075 - val_loss: 2.2768e-04 - val_mae: 0.0149
    Epoch 190/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.6029e-05 - mae: 0.0074 - val_loss: 2.3228e-04 - val_mae: 0.0156
    Epoch 191/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.6284e-05 - mae: 0.0075 - val_loss: 2.3720e-04 - val_mae: 0.0153
    Epoch 192/500
    43/43 [==============================] - ETA: 0s - loss: 5.5731e-05 - mae: 0.0074INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 5.5731e-05 - mae: 0.0074 - val_loss: 2.2054e-04 - val_mae: 0.0149
    Epoch 193/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.6154e-05 - mae: 0.0074 - val_loss: 2.4649e-04 - val_mae: 0.0165
    Epoch 194/500
    42/43 [============================>.] - ETA: 0s - loss: 5.4952e-05 - mae: 0.0074INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 5.5185e-05 - mae: 0.0074 - val_loss: 2.1657e-04 - val_mae: 0.0146
    Epoch 195/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.5173e-05 - mae: 0.0073 - val_loss: 2.3962e-04 - val_mae: 0.0162
    Epoch 196/500
    38/43 [=========================>....] - ETA: 0s - loss: 5.2067e-05 - mae: 0.0071INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.5333e-05 - mae: 0.0074 - val_loss: 2.1618e-04 - val_mae: 0.0147
    Epoch 197/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.4878e-05 - mae: 0.0074 - val_loss: 2.1966e-04 - val_mae: 0.0150
    Epoch 198/500
    35/43 [=======================>......] - ETA: 0s - loss: 5.0174e-05 - mae: 0.0069INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.5135e-05 - mae: 0.0074 - val_loss: 2.1453e-04 - val_mae: 0.0147
    Epoch 199/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.4115e-05 - mae: 0.0073 - val_loss: 2.4200e-04 - val_mae: 0.0156
    Epoch 200/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.6442e-05 - mae: 0.0075 - val_loss: 2.5479e-04 - val_mae: 0.0162
    Epoch 201/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.7835e-05 - mae: 0.0075 - val_loss: 2.5346e-04 - val_mae: 0.0171
    Epoch 202/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.5962e-05 - mae: 0.0074 - val_loss: 2.2102e-04 - val_mae: 0.0147
    Epoch 203/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.4559e-05 - mae: 0.0073 - val_loss: 2.1696e-04 - val_mae: 0.0145
    Epoch 204/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.2963e-05 - mae: 0.0072 - val_loss: 2.2291e-04 - val_mae: 0.0155
    Epoch 205/500
    38/43 [=========================>....] - ETA: 0s - loss: 5.0813e-05 - mae: 0.0070INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.2864e-05 - mae: 0.0072 - val_loss: 2.0807e-04 - val_mae: 0.0145
    Epoch 206/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.2336e-05 - mae: 0.0072 - val_loss: 2.3155e-04 - val_mae: 0.0152
    Epoch 207/500
    40/43 [==========================>...] - ETA: 0s - loss: 5.0744e-05 - mae: 0.0070INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 5.4439e-05 - mae: 0.0073 - val_loss: 2.0692e-04 - val_mae: 0.0145
    Epoch 208/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.2436e-05 - mae: 0.0072 - val_loss: 2.1826e-04 - val_mae: 0.0153
    Epoch 209/500
    40/43 [==========================>...] - ETA: 0s - loss: 5.0810e-05 - mae: 0.0070INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 5.1949e-05 - mae: 0.0071 - val_loss: 2.0280e-04 - val_mae: 0.0142
    Epoch 210/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.1746e-05 - mae: 0.0071 - val_loss: 2.1789e-04 - val_mae: 0.0146
    Epoch 211/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.1654e-05 - mae: 0.0071 - val_loss: 2.0736e-04 - val_mae: 0.0142
    Epoch 212/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.0787e-05 - mae: 0.0070 - val_loss: 2.0975e-04 - val_mae: 0.0143
    Epoch 213/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.6345e-05 - mae: 0.0066INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 21ms/step - loss: 5.2460e-05 - mae: 0.0072 - val_loss: 1.9844e-04 - val_mae: 0.0139
    Epoch 214/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.1495e-05 - mae: 0.0071 - val_loss: 2.3525e-04 - val_mae: 0.0155
    Epoch 215/500
    42/43 [============================>.] - ETA: 0s - loss: 5.0087e-05 - mae: 0.0070INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 5.0529e-05 - mae: 0.0070 - val_loss: 1.9611e-04 - val_mae: 0.0139
    Epoch 216/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.0034e-05 - mae: 0.0070 - val_loss: 2.0620e-04 - val_mae: 0.0142
    Epoch 217/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.1606e-05 - mae: 0.0071 - val_loss: 1.9775e-04 - val_mae: 0.0142
    Epoch 218/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.9690e-05 - mae: 0.0070 - val_loss: 2.0010e-04 - val_mae: 0.0139
    Epoch 219/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.6576e-05 - mae: 0.0067INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 5.1369e-05 - mae: 0.0071 - val_loss: 1.9407e-04 - val_mae: 0.0139
    Epoch 220/500
    41/43 [===========================>..] - ETA: 0s - loss: 4.8645e-05 - mae: 0.0069INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.9852e-05 - mae: 0.0070 - val_loss: 1.9206e-04 - val_mae: 0.0137
    Epoch 221/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.9115e-05 - mae: 0.0069 - val_loss: 1.9779e-04 - val_mae: 0.0138
    Epoch 222/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.6076e-05 - mae: 0.0066INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.9118e-05 - mae: 0.0069 - val_loss: 1.9028e-04 - val_mae: 0.0136
    Epoch 223/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.8684e-05 - mae: 0.0069 - val_loss: 2.8986e-04 - val_mae: 0.0194
    Epoch 224/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.6005e-05 - mae: 0.0067INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.9184e-05 - mae: 0.0070 - val_loss: 1.8973e-04 - val_mae: 0.0135
    Epoch 225/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.5212e-05 - mae: 0.0065INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.8244e-05 - mae: 0.0069 - val_loss: 1.8841e-04 - val_mae: 0.0135
    Epoch 226/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.8445e-05 - mae: 0.0069 - val_loss: 1.9124e-04 - val_mae: 0.0140
    Epoch 227/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.2274e-05 - mae: 0.0072 - val_loss: 2.4963e-04 - val_mae: 0.0175
    Epoch 228/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.6272e-05 - mae: 0.0068INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.8266e-05 - mae: 0.0069 - val_loss: 1.8545e-04 - val_mae: 0.0134
    Epoch 229/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.3813e-05 - mae: 0.0065INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.7855e-05 - mae: 0.0068 - val_loss: 1.8457e-04 - val_mae: 0.0134
    Epoch 230/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.7975e-05 - mae: 0.0069 - val_loss: 1.9418e-04 - val_mae: 0.0143
    Epoch 231/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.8218e-05 - mae: 0.0069 - val_loss: 1.9631e-04 - val_mae: 0.0138
    Epoch 232/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.8904e-05 - mae: 0.0069 - val_loss: 1.9952e-04 - val_mae: 0.0147
    Epoch 233/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.0943e-05 - mae: 0.0071 - val_loss: 1.8608e-04 - val_mae: 0.0134
    Epoch 234/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.6741e-05 - mae: 0.0067 - val_loss: 1.8568e-04 - val_mae: 0.0138
    Epoch 235/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.7015e-05 - mae: 0.0068 - val_loss: 2.4237e-04 - val_mae: 0.0161
    Epoch 236/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.6588e-05 - mae: 0.0068 - val_loss: 2.0925e-04 - val_mae: 0.0145
    Epoch 237/500
    38/43 [=========================>....] - ETA: 0s - loss: 4.3539e-05 - mae: 0.0065INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.6083e-05 - mae: 0.0067 - val_loss: 1.8242e-04 - val_mae: 0.0136
    Epoch 238/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.2540e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.6371e-05 - mae: 0.0067 - val_loss: 1.7785e-04 - val_mae: 0.0132
    Epoch 239/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.5380e-05 - mae: 0.0066 - val_loss: 1.7804e-04 - val_mae: 0.0131
    Epoch 240/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.4972e-05 - mae: 0.0066INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.5925e-05 - mae: 0.0067 - val_loss: 1.7653e-04 - val_mae: 0.0132
    Epoch 241/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.7692e-05 - mae: 0.0068 - val_loss: 3.1488e-04 - val_mae: 0.0208
    Epoch 242/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.6353e-05 - mae: 0.0068 - val_loss: 1.7881e-04 - val_mae: 0.0131
    Epoch 243/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.5524e-05 - mae: 0.0067 - val_loss: 2.1724e-04 - val_mae: 0.0150
    Epoch 244/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.6898e-05 - mae: 0.0068 - val_loss: 1.8724e-04 - val_mae: 0.0135
    Epoch 245/500
    42/43 [============================>.] - ETA: 0s - loss: 4.3828e-05 - mae: 0.0065INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.4717e-05 - mae: 0.0066 - val_loss: 1.7410e-04 - val_mae: 0.0132
    Epoch 246/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.4450e-05 - mae: 0.0065INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 24ms/step - loss: 4.6244e-05 - mae: 0.0067 - val_loss: 1.7233e-04 - val_mae: 0.0129
    Epoch 247/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.4736e-05 - mae: 0.0066 - val_loss: 1.7298e-04 - val_mae: 0.0131
    Epoch 248/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.5378e-05 - mae: 0.0067 - val_loss: 2.2261e-04 - val_mae: 0.0164
    Epoch 249/500
    35/43 [=======================>......] - ETA: 0s - loss: 4.2324e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.7760e-05 - mae: 0.0069 - val_loss: 1.7021e-04 - val_mae: 0.0129
    Epoch 250/500
    38/43 [=========================>....] - ETA: 0s - loss: 4.2741e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.4991e-05 - mae: 0.0066 - val_loss: 1.7009e-04 - val_mae: 0.0128
    Epoch 251/500
    37/43 [========================>.....] - ETA: 0s - loss: 4.1901e-05 - mae: 0.0063INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.4540e-05 - mae: 0.0066 - val_loss: 1.6909e-04 - val_mae: 0.0129
    Epoch 252/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.2471e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.3461e-05 - mae: 0.0065 - val_loss: 1.6847e-04 - val_mae: 0.0128
    Epoch 253/500
    38/43 [=========================>....] - ETA: 0s - loss: 4.1956e-05 - mae: 0.0063INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.3836e-05 - mae: 0.0065 - val_loss: 1.6785e-04 - val_mae: 0.0128
    Epoch 254/500
    36/43 [========================>.....] - ETA: 0s - loss: 4.0707e-05 - mae: 0.0062INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 4.3557e-05 - mae: 0.0065 - val_loss: 1.6771e-04 - val_mae: 0.0127
    Epoch 255/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.3178e-05 - mae: 0.0065 - val_loss: 1.8856e-04 - val_mae: 0.0145
    Epoch 256/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.3389e-05 - mae: 0.0065 - val_loss: 1.8161e-04 - val_mae: 0.0133
    Epoch 257/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.3043e-05 - mae: 0.0065 - val_loss: 1.9741e-04 - val_mae: 0.0141
    Epoch 258/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.2060e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 24ms/step - loss: 4.2862e-05 - mae: 0.0065 - val_loss: 1.6548e-04 - val_mae: 0.0126
    Epoch 259/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.2761e-05 - mae: 0.0064 - val_loss: 1.7941e-04 - val_mae: 0.0132
    Epoch 260/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.3201e-05 - mae: 0.0065 - val_loss: 2.1698e-04 - val_mae: 0.0152
    Epoch 261/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.2800e-05 - mae: 0.0065 - val_loss: 1.7083e-04 - val_mae: 0.0133
    Epoch 262/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.2621e-05 - mae: 0.0065 - val_loss: 1.7961e-04 - val_mae: 0.0140
    Epoch 263/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.3345e-05 - mae: 0.0065 - val_loss: 1.7090e-04 - val_mae: 0.0134
    Epoch 264/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.1979e-05 - mae: 0.0064 - val_loss: 1.8723e-04 - val_mae: 0.0137
    Epoch 265/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.1734e-05 - mae: 0.0064 - val_loss: 1.7145e-04 - val_mae: 0.0135
    Epoch 266/500
    38/43 [=========================>....] - ETA: 0s - loss: 4.0149e-05 - mae: 0.0062INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.1811e-05 - mae: 0.0064 - val_loss: 1.6547e-04 - val_mae: 0.0130
    Epoch 267/500
    38/43 [=========================>....] - ETA: 0s - loss: 4.0005e-05 - mae: 0.0062INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.3860e-05 - mae: 0.0065 - val_loss: 1.6489e-04 - val_mae: 0.0126
    Epoch 268/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.0888e-05 - mae: 0.0063INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.1414e-05 - mae: 0.0063 - val_loss: 1.6220e-04 - val_mae: 0.0128
    Epoch 269/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.1418e-05 - mae: 0.0064 - val_loss: 1.6740e-04 - val_mae: 0.0127
    Epoch 270/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.1235e-05 - mae: 0.0063 - val_loss: 2.5506e-04 - val_mae: 0.0171
    Epoch 271/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.6701e-05 - mae: 0.0068 - val_loss: 1.7803e-04 - val_mae: 0.0141
    Epoch 272/500
    36/43 [========================>.....] - ETA: 0s - loss: 3.9013e-05 - mae: 0.0061INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.1016e-05 - mae: 0.0063 - val_loss: 1.5786e-04 - val_mae: 0.0123
    Epoch 273/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.0490e-05 - mae: 0.0063 - val_loss: 1.5896e-04 - val_mae: 0.0123
    Epoch 274/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.0528e-05 - mae: 0.0063 - val_loss: 1.6885e-04 - val_mae: 0.0135
    Epoch 275/500
    36/43 [========================>.....] - ETA: 0s - loss: 3.7103e-05 - mae: 0.0060INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 4.0654e-05 - mae: 0.0063 - val_loss: 1.5503e-04 - val_mae: 0.0123
    Epoch 276/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.1125e-05 - mae: 0.0063 - val_loss: 1.9318e-04 - val_mae: 0.0141
    Epoch 277/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.0960e-05 - mae: 0.0063 - val_loss: 1.6986e-04 - val_mae: 0.0129
    Epoch 278/500
    41/43 [===========================>..] - ETA: 0s - loss: 4.1982e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.2215e-05 - mae: 0.0064 - val_loss: 1.5381e-04 - val_mae: 0.0121
    Epoch 279/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.9915e-05 - mae: 0.0062 - val_loss: 1.6034e-04 - val_mae: 0.0129
    Epoch 280/500
    41/43 [===========================>..] - ETA: 0s - loss: 4.1654e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 4.2074e-05 - mae: 0.0064 - val_loss: 1.5377e-04 - val_mae: 0.0121
    Epoch 281/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.9668e-05 - mae: 0.0062 - val_loss: 1.5451e-04 - val_mae: 0.0121
    Epoch 282/500
    43/43 [==============================] - ETA: 0s - loss: 3.9567e-05 - mae: 0.0062INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 3.9567e-05 - mae: 0.0062 - val_loss: 1.5135e-04 - val_mae: 0.0121
    Epoch 283/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.9671e-05 - mae: 0.0062 - val_loss: 1.8480e-04 - val_mae: 0.0137
    Epoch 284/500
    36/43 [========================>.....] - ETA: 0s - loss: 3.7429e-05 - mae: 0.0059INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 3.9648e-05 - mae: 0.0062 - val_loss: 1.5035e-04 - val_mae: 0.0121
    Epoch 285/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.8714e-05 - mae: 0.0061INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 3.9109e-05 - mae: 0.0062 - val_loss: 1.5016e-04 - val_mae: 0.0121
    Epoch 286/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.8911e-05 - mae: 0.0061 - val_loss: 1.5180e-04 - val_mae: 0.0123
    Epoch 287/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.9051e-05 - mae: 0.0061 - val_loss: 1.6461e-04 - val_mae: 0.0134
    Epoch 288/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9788e-05 - mae: 0.0062 - val_loss: 1.6436e-04 - val_mae: 0.0127
    Epoch 289/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9013e-05 - mae: 0.0061 - val_loss: 1.5725e-04 - val_mae: 0.0123
    Epoch 290/500
    43/43 [==============================] - 1s 20ms/step - loss: 4.0616e-05 - mae: 0.0063 - val_loss: 1.5058e-04 - val_mae: 0.0120
    Epoch 291/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.8796e-05 - mae: 0.0061 - val_loss: 1.6884e-04 - val_mae: 0.0138
    Epoch 292/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.8471e-05 - mae: 0.0061 - val_loss: 1.6747e-04 - val_mae: 0.0129
    Epoch 293/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.8167e-05 - mae: 0.0061 - val_loss: 1.8956e-04 - val_mae: 0.0141
    Epoch 294/500
    38/43 [=========================>....] - ETA: 0s - loss: 3.6791e-05 - mae: 0.0059INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 25ms/step - loss: 3.8548e-05 - mae: 0.0061 - val_loss: 1.5008e-04 - val_mae: 0.0124
    Epoch 295/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.8319e-05 - mae: 0.0061 - val_loss: 1.5023e-04 - val_mae: 0.0124
    Epoch 296/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.8271e-05 - mae: 0.0061 - val_loss: 1.6934e-04 - val_mae: 0.0130
    Epoch 297/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.8273e-05 - mae: 0.0061 - val_loss: 1.5354e-04 - val_mae: 0.0127
    Epoch 298/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.6973e-05 - mae: 0.0060INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 24ms/step - loss: 3.7909e-05 - mae: 0.0061 - val_loss: 1.4640e-04 - val_mae: 0.0121
    Epoch 299/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.8310e-05 - mae: 0.0061 - val_loss: 1.7235e-04 - val_mae: 0.0132
    Epoch 300/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.8895e-05 - mae: 0.0062 - val_loss: 1.9626e-04 - val_mae: 0.0145
    Epoch 301/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.9352e-05 - mae: 0.0062 - val_loss: 1.7727e-04 - val_mae: 0.0145
    Epoch 302/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.7474e-05 - mae: 0.0061 - val_loss: 1.7475e-04 - val_mae: 0.0133
    Epoch 303/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.8758e-05 - mae: 0.0062 - val_loss: 1.4856e-04 - val_mae: 0.0119
    Epoch 304/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.6873e-05 - mae: 0.0060 - val_loss: 1.4798e-04 - val_mae: 0.0124
    Epoch 305/500
    38/43 [=========================>....] - ETA: 0s - loss: 3.5094e-05 - mae: 0.0058INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 22ms/step - loss: 3.7264e-05 - mae: 0.0060 - val_loss: 1.4103e-04 - val_mae: 0.0116
    Epoch 306/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.7065e-05 - mae: 0.0060 - val_loss: 1.4987e-04 - val_mae: 0.0126
    Epoch 307/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.8486e-05 - mae: 0.0061 - val_loss: 1.4259e-04 - val_mae: 0.0116
    Epoch 308/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.7701e-05 - mae: 0.0060 - val_loss: 1.4293e-04 - val_mae: 0.0116
    Epoch 309/500
    37/43 [========================>.....] - ETA: 0s - loss: 3.4249e-05 - mae: 0.0057INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 3.6623e-05 - mae: 0.0059 - val_loss: 1.3906e-04 - val_mae: 0.0115
    Epoch 310/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.6465e-05 - mae: 0.0060 - val_loss: 1.4504e-04 - val_mae: 0.0118
    Epoch 311/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.7704e-05 - mae: 0.0060 - val_loss: 1.6691e-04 - val_mae: 0.0130
    Epoch 312/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.6789e-05 - mae: 0.0060 - val_loss: 1.5623e-04 - val_mae: 0.0124
    Epoch 313/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.6781e-05 - mae: 0.0060 - val_loss: 1.6586e-04 - val_mae: 0.0129
    Epoch 314/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.8738e-05 - mae: 0.0061 - val_loss: 1.6767e-04 - val_mae: 0.0140
    Epoch 315/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.6379e-05 - mae: 0.0060 - val_loss: 1.4065e-04 - val_mae: 0.0119
    Epoch 316/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.7097e-05 - mae: 0.0060 - val_loss: 1.4240e-04 - val_mae: 0.0121
    Epoch 317/500
    41/43 [===========================>..] - ETA: 0s - loss: 3.5428e-05 - mae: 0.0059INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 3.5844e-05 - mae: 0.0059 - val_loss: 1.3606e-04 - val_mae: 0.0114
    Epoch 318/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.5866e-05 - mae: 0.0059 - val_loss: 1.4084e-04 - val_mae: 0.0116
    Epoch 319/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.7152e-05 - mae: 0.0060 - val_loss: 1.3905e-04 - val_mae: 0.0119
    Epoch 320/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.6133e-05 - mae: 0.0059 - val_loss: 1.8015e-04 - val_mae: 0.0138
    Epoch 321/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.6192e-05 - mae: 0.0059 - val_loss: 1.3919e-04 - val_mae: 0.0119
    Epoch 322/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.8338e-05 - mae: 0.0061 - val_loss: 1.8074e-04 - val_mae: 0.0138
    Epoch 323/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5831e-05 - mae: 0.0059 - val_loss: 1.4138e-04 - val_mae: 0.0121
    Epoch 324/500
    38/43 [=========================>....] - ETA: 0s - loss: 3.3378e-05 - mae: 0.0056INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 3.5383e-05 - mae: 0.0059 - val_loss: 1.3379e-04 - val_mae: 0.0113
    Epoch 325/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.5147e-05 - mae: 0.0058 - val_loss: 1.3517e-04 - val_mae: 0.0113
    Epoch 326/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.4972e-05 - mae: 0.0058 - val_loss: 1.3448e-04 - val_mae: 0.0115
    Epoch 327/500
    41/43 [===========================>..] - ETA: 0s - loss: 3.5128e-05 - mae: 0.0058INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 23ms/step - loss: 3.5336e-05 - mae: 0.0058 - val_loss: 1.3239e-04 - val_mae: 0.0113
    Epoch 328/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.5138e-05 - mae: 0.0058 - val_loss: 1.3517e-04 - val_mae: 0.0113
    Epoch 329/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5030e-05 - mae: 0.0058 - val_loss: 1.3508e-04 - val_mae: 0.0117
    Epoch 330/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.6038e-05 - mae: 0.0059 - val_loss: 1.4901e-04 - val_mae: 0.0121
    Epoch 331/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5270e-05 - mae: 0.0058 - val_loss: 1.3692e-04 - val_mae: 0.0114
    Epoch 332/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.5237e-05 - mae: 0.0058 - val_loss: 1.3874e-04 - val_mae: 0.0121
    Epoch 333/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.4797e-05 - mae: 0.0058 - val_loss: 1.3375e-04 - val_mae: 0.0116
    Epoch 334/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5912e-05 - mae: 0.0059 - val_loss: 1.7908e-04 - val_mae: 0.0138
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
