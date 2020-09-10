---
title: "Project: Part 3 Dense Forecast with Keras"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Deep Learning
---
**This notebook is part 1 of a multi-phase project of building a quantitative trading algorithm. This notebook starts with the more simple models which will be compared to more complex deep learning models. This notebook's last image shows the final results of all models tested.**

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
    Epoch 4/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.2009 - mae: 0.5512
    Epoch 5/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2008 - mae: 0.5511
    Epoch 6/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2007 - mae: 0.5510
    Epoch 7/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.2006 - mae: 0.5508
    Epoch 8/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2005 - mae: 0.5506
    Epoch 9/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2003 - mae: 0.5504
    Epoch 10/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.2002 - mae: 0.5502
    Epoch 11/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.2000 - mae: 0.5499
    Epoch 12/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1998 - mae: 0.5496
    Epoch 13/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1996 - mae: 0.5493
    Epoch 14/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1994 - mae: 0.5489
    Epoch 15/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1991 - mae: 0.5485
    Epoch 16/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1988 - mae: 0.5481
    Epoch 17/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1984 - mae: 0.5475
    Epoch 18/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1981 - mae: 0.5470
    Epoch 19/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1976 - mae: 0.5463
    Epoch 20/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1972 - mae: 0.5456
    Epoch 21/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1966 - mae: 0.5448
    Epoch 22/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1960 - mae: 0.5439
    Epoch 23/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1953 - mae: 0.5428
    Epoch 24/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1946 - mae: 0.5417
    Epoch 25/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1937 - mae: 0.5404
    Epoch 26/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1928 - mae: 0.5390
    Epoch 27/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1917 - mae: 0.5374
    Epoch 28/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1906 - mae: 0.5356
    Epoch 29/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.1893 - mae: 0.5335
    Epoch 30/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1878 - mae: 0.5313
    Epoch 31/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1861 - mae: 0.5287
    Epoch 32/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1843 - mae: 0.5259
    Epoch 33/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1823 - mae: 0.5227
    Epoch 34/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1800 - mae: 0.5192
    Epoch 35/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1775 - mae: 0.5152
    Epoch 36/100
    43/43 [==============================] - 0s 11ms/step - loss: 0.1747 - mae: 0.5108
    Epoch 37/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1716 - mae: 0.5058
    Epoch 38/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1682 - mae: 0.5003
    Epoch 39/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1644 - mae: 0.4942
    Epoch 40/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1602 - mae: 0.4874
    Epoch 41/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1555 - mae: 0.4798
    Epoch 42/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1505 - mae: 0.4714
    Epoch 43/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1450 - mae: 0.4621
    Epoch 44/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1389 - mae: 0.4518
    Epoch 45/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1324 - mae: 0.4405
    Epoch 46/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1254 - mae: 0.4280
    Epoch 47/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1178 - mae: 0.4144
    Epoch 48/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1097 - mae: 0.3995
    Epoch 49/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.1012 - mae: 0.3831
    Epoch 50/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0922 - mae: 0.3653
    Epoch 51/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0829 - mae: 0.3459
    Epoch 52/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0734 - mae: 0.3250
    Epoch 53/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0639 - mae: 0.3027
    Epoch 54/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0545 - mae: 0.2793
    Epoch 55/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0463 - mae: 0.2571
    Epoch 56/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0394 - mae: 0.2367
    Epoch 57/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0330 - mae: 0.2162
    Epoch 58/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0269 - mae: 0.1947
    Epoch 59/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0213 - mae: 0.1725
    Epoch 60/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0162 - mae: 0.1500
    Epoch 61/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0119 - mae: 0.1277
    Epoch 62/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0083 - mae: 0.1062
    Epoch 63/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0056 - mae: 0.0864
    Epoch 64/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0037 - mae: 0.0691
    Epoch 65/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0024 - mae: 0.0553
    Epoch 66/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0017 - mae: 0.0457
    Epoch 67/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0014 - mae: 0.0404
    Epoch 68/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0012 - mae: 0.0381
    Epoch 69/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0011 - mae: 0.0373
    Epoch 70/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0011 - mae: 0.0369
    Epoch 71/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0010 - mae: 0.0364
    Epoch 72/100
    43/43 [==============================] - 1s 13ms/step - loss: 9.7454e-04 - mae: 0.0357
    Epoch 73/100
    43/43 [==============================] - 1s 12ms/step - loss: 9.0331e-04 - mae: 0.0345
    Epoch 74/100
    43/43 [==============================] - 1s 14ms/step - loss: 7.6208e-04 - mae: 0.0316
    Epoch 75/100
    43/43 [==============================] - 1s 13ms/step - loss: 6.0085e-04 - mae: 0.0282
    Epoch 76/100
    43/43 [==============================] - 1s 13ms/step - loss: 4.2562e-04 - mae: 0.0238
    Epoch 77/100
    43/43 [==============================] - 1s 14ms/step - loss: 2.9549e-04 - mae: 0.0197
    Epoch 78/100
    43/43 [==============================] - 1s 12ms/step - loss: 2.2111e-04 - mae: 0.0165
    Epoch 79/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.9089e-04 - mae: 0.0145
    Epoch 80/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.8168e-04 - mae: 0.0137
    Epoch 81/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.7846e-04 - mae: 0.0135
    Epoch 82/100
    43/43 [==============================] - 1s 12ms/step - loss: 1.7742e-04 - mae: 0.0135
    Epoch 83/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.7492e-04 - mae: 0.0134
    Epoch 84/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.7351e-04 - mae: 0.0134
    Epoch 85/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.7189e-04 - mae: 0.0133
    Epoch 86/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.7037e-04 - mae: 0.0132
    Epoch 87/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.6769e-04 - mae: 0.0131
    Epoch 88/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.6421e-04 - mae: 0.0130
    Epoch 89/100
    43/43 [==============================] - 1s 12ms/step - loss: 1.6325e-04 - mae: 0.0129
    Epoch 90/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.6127e-04 - mae: 0.0128
    Epoch 91/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.5731e-04 - mae: 0.0126
    Epoch 92/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.5698e-04 - mae: 0.0126
    Epoch 93/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.5572e-04 - mae: 0.0127
    Epoch 94/100
    43/43 [==============================] - 1s 12ms/step - loss: 1.5131e-04 - mae: 0.0125
    Epoch 95/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.4620e-04 - mae: 0.0122
    Epoch 96/100
    43/43 [==============================] - 1s 12ms/step - loss: 1.5027e-04 - mae: 0.0124
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
    Epoch 5/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.0527e-04 - mae: 0.0202 - val_loss: 0.0128 - val_mae: 0.1549
    Epoch 6/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.9926e-04 - mae: 0.0151 - val_loss: 8.0172e-04 - val_mae: 0.0340
    Epoch 7/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.2152e-04 - mae: 0.0194 - val_loss: 0.0643 - val_mae: 0.3531
    Epoch 8/500
    43/43 [==============================] - 1s 17ms/step - loss: 0.0014 - mae: 0.0300 - val_loss: 0.0434 - val_mae: 0.2896
    Epoch 9/500
    43/43 [==============================] - 1s 18ms/step - loss: 0.0010 - mae: 0.0285 - val_loss: 0.0431 - val_mae: 0.2887
    Epoch 10/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6672e-04 - mae: 0.0170 - val_loss: 0.0011 - val_mae: 0.0353
    Epoch 11/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.7195e-04 - mae: 0.0138 - val_loss: 0.0037 - val_mae: 0.0793
    Epoch 12/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.7421e-04 - mae: 0.0139 - val_loss: 0.0047 - val_mae: 0.0920
    Epoch 13/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.8074e-04 - mae: 0.0140 - val_loss: 0.0046 - val_mae: 0.0907
    Epoch 14/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.5683e-04 - mae: 0.0154 - val_loss: 0.0174 - val_mae: 0.1819
    Epoch 15/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.9480e-04 - mae: 0.0186 - val_loss: 0.0474 - val_mae: 0.3034
    Epoch 16/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.1231e-04 - mae: 0.0189 - val_loss: 0.0044 - val_mae: 0.0886
    Epoch 17/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.9205e-04 - mae: 0.0178 - val_loss: 0.0355 - val_mae: 0.2618
    Epoch 18/500
    43/43 [==============================] - 1s 18ms/step - loss: 7.8858e-04 - mae: 0.0240 - val_loss: 0.0717 - val_mae: 0.3736
    Epoch 19/500
    37/43 [========================>.....] - ETA: 0s - loss: 4.0185e-04 - mae: 0.0210INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 28ms/step - loss: 3.8366e-04 - mae: 0.0205 - val_loss: 6.8575e-04 - val_mae: 0.0312
    Epoch 20/500
    43/43 [==============================] - ETA: 0s - loss: 1.2275e-04 - mae: 0.0115INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 1.2275e-04 - mae: 0.0115 - val_loss: 4.0509e-04 - val_mae: 0.0215
    Epoch 21/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.9564e-04 - mae: 0.0137 - val_loss: 0.0114 - val_mae: 0.1468
    Epoch 22/500
    43/43 [==============================] - 1s 16ms/step - loss: 1.4554e-04 - mae: 0.0128 - val_loss: 0.0013 - val_mae: 0.0420
    Epoch 23/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.0419e-04 - mae: 0.0195 - val_loss: 0.0635 - val_mae: 0.3510
    Epoch 24/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.5263e-04 - mae: 0.0252 - val_loss: 0.0127 - val_mae: 0.1541
    Epoch 25/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.3583e-04 - mae: 0.0151 - val_loss: 0.0110 - val_mae: 0.1447
    Epoch 26/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.2320e-04 - mae: 0.0209 - val_loss: 0.0285 - val_mae: 0.2348
    Epoch 27/500
    36/43 [========================>.....] - ETA: 0s - loss: 1.9247e-04 - mae: 0.0148INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 28ms/step - loss: 1.8275e-04 - mae: 0.0144 - val_loss: 3.5815e-04 - val_mae: 0.0196
    Epoch 28/500
    43/43 [==============================] - 1s 16ms/step - loss: 1.5982e-04 - mae: 0.0127 - val_loss: 0.0031 - val_mae: 0.0741
    Epoch 29/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.2635e-04 - mae: 0.0117 - val_loss: 0.0023 - val_mae: 0.0632
    Epoch 30/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.8207e-04 - mae: 0.0198 - val_loss: 0.0235 - val_mae: 0.2131
    Epoch 31/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2992e-04 - mae: 0.0157 - val_loss: 0.0076 - val_mae: 0.1190
    Epoch 32/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.4958e-04 - mae: 0.0127 - val_loss: 0.0024 - val_mae: 0.0640
    Epoch 33/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.3009e-04 - mae: 0.0119 - val_loss: 0.0024 - val_mae: 0.0647
    Epoch 34/500
    36/43 [========================>.....] - ETA: 0s - loss: 9.8502e-05 - mae: 0.0101INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 1.0103e-04 - mae: 0.0104 - val_loss: 3.2230e-04 - val_mae: 0.0184
    Epoch 35/500
    43/43 [==============================] - 1s 17ms/step - loss: 9.5329e-05 - mae: 0.0098 - val_loss: 9.3711e-04 - val_mae: 0.0353
    Epoch 36/500
    43/43 [==============================] - 1s 17ms/step - loss: 9.5531e-05 - mae: 0.0099 - val_loss: 0.0011 - val_mae: 0.0423
    Epoch 37/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.0171e-04 - mae: 0.0156 - val_loss: 0.0621 - val_mae: 0.3473
    Epoch 38/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.4138e-04 - mae: 0.0216 - val_loss: 0.0073 - val_mae: 0.1178
    Epoch 39/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.8919e-04 - mae: 0.0141 - val_loss: 0.0103 - val_mae: 0.1402
    Epoch 40/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.5158e-04 - mae: 0.0150 - val_loss: 0.0244 - val_mae: 0.2173
    Epoch 41/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.7997e-04 - mae: 0.0140 - val_loss: 0.0020 - val_mae: 0.0579
    Epoch 42/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.9304e-04 - mae: 0.0127 - val_loss: 0.0180 - val_mae: 0.1857
    Epoch 43/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.3784e-04 - mae: 0.0124 - val_loss: 3.3682e-04 - val_mae: 0.0177
    Epoch 44/500
    43/43 [==============================] - 1s 16ms/step - loss: 1.7533e-04 - mae: 0.0122 - val_loss: 0.0114 - val_mae: 0.1476
    Epoch 45/500
    43/43 [==============================] - ETA: 0s - loss: 1.1406e-04 - mae: 0.0114INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 28ms/step - loss: 1.1406e-04 - mae: 0.0114 - val_loss: 3.1410e-04 - val_mae: 0.0196
    Epoch 46/500
    43/43 [==============================] - 1s 15ms/step - loss: 9.4102e-04 - mae: 0.0208 - val_loss: 0.0644 - val_mae: 0.3533
    Epoch 47/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.7562e-04 - mae: 0.0205 - val_loss: 0.0034 - val_mae: 0.0766
    Epoch 48/500
    37/43 [========================>.....] - ETA: 0s - loss: 1.1360e-04 - mae: 0.0108INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 30ms/step - loss: 1.1491e-04 - mae: 0.0110 - val_loss: 2.8048e-04 - val_mae: 0.0169
    Epoch 49/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.9222e-04 - mae: 0.0200 - val_loss: 0.0072 - val_mae: 0.1169
    Epoch 50/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.5612e-04 - mae: 0.0130 - val_loss: 0.0019 - val_mae: 0.0572
    Epoch 51/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.8285e-05 - mae: 0.0097 - val_loss: 0.0014 - val_mae: 0.0483
    Epoch 52/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.2319e-05 - mae: 0.0092 - val_loss: 6.1766e-04 - val_mae: 0.0308
    Epoch 53/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.6749e-04 - mae: 0.0118 - val_loss: 0.0066 - val_mae: 0.1113
    Epoch 54/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1897e-04 - mae: 0.0114 - val_loss: 0.0031 - val_mae: 0.0750
    Epoch 55/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.0451e-04 - mae: 0.0105 - val_loss: 0.0034 - val_mae: 0.0791
    Epoch 56/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.2336e-04 - mae: 0.0110 - val_loss: 0.0055 - val_mae: 0.1014
    Epoch 57/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.7549e-05 - mae: 0.0099 - val_loss: 5.4237e-04 - val_mae: 0.0286
    Epoch 58/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.4162e-04 - mae: 0.0154 - val_loss: 0.0584 - val_mae: 0.3370
    Epoch 59/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.2605e-04 - mae: 0.0188 - val_loss: 3.7197e-04 - val_mae: 0.0220
    Epoch 60/500
    43/43 [==============================] - 1s 18ms/step - loss: 8.3775e-05 - mae: 0.0096 - val_loss: 6.8746e-04 - val_mae: 0.0330
    Epoch 61/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.3107e-04 - mae: 0.0158 - val_loss: 0.0275 - val_mae: 0.2313
    Epoch 62/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.6381e-04 - mae: 0.0132 - val_loss: 3.6612e-04 - val_mae: 0.0188
    Epoch 63/500
    42/43 [============================>.] - ETA: 0s - loss: 6.9676e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 30ms/step - loss: 7.0046e-05 - mae: 0.0086 - val_loss: 2.4549e-04 - val_mae: 0.0161
    Epoch 64/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2372e-04 - mae: 0.0128 - val_loss: 0.0175 - val_mae: 0.1836
    Epoch 65/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.7910e-04 - mae: 0.0167 - val_loss: 0.0118 - val_mae: 0.1499
    Epoch 66/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.3506e-04 - mae: 0.0123 - val_loss: 0.0015 - val_mae: 0.0495
    Epoch 67/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.0164e-04 - mae: 0.0101 - val_loss: 0.0041 - val_mae: 0.0872
    Epoch 68/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.0919e-05 - mae: 0.0093 - val_loss: 3.3062e-04 - val_mae: 0.0210
    Epoch 69/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.3207e-04 - mae: 0.0106 - val_loss: 0.0091 - val_mae: 0.1316
    Epoch 70/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.2811e-04 - mae: 0.0115 - val_loss: 0.0064 - val_mae: 0.1096
    Epoch 71/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.4479e-04 - mae: 0.0157 - val_loss: 0.0346 - val_mae: 0.2588
    Epoch 72/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.9486e-04 - mae: 0.0179 - val_loss: 0.0040 - val_mae: 0.0868
    Epoch 73/500
    39/43 [==========================>...] - ETA: 0s - loss: 9.0584e-05 - mae: 0.0099INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 29ms/step - loss: 9.0363e-05 - mae: 0.0099 - val_loss: 2.4221e-04 - val_mae: 0.0167
    Epoch 74/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.2586e-04 - mae: 0.0103 - val_loss: 0.0062 - val_mae: 0.1084
    Epoch 75/500
    43/43 [==============================] - 1s 18ms/step - loss: 8.2119e-05 - mae: 0.0096 - val_loss: 2.4618e-04 - val_mae: 0.0171
    Epoch 76/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.3297e-05 - mae: 0.0086 - val_loss: 0.0027 - val_mae: 0.0698
    Epoch 77/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.9082e-04 - mae: 0.0163 - val_loss: 0.0272 - val_mae: 0.2303
    Epoch 78/500
    43/43 [==============================] - 1s 17ms/step - loss: 0.0015 - mae: 0.0317 - val_loss: 2.6856e-04 - val_mae: 0.0174
    Epoch 79/500
    43/43 [==============================] - ETA: 0s - loss: 3.6879e-04 - mae: 0.0172INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 33ms/step - loss: 3.6879e-04 - mae: 0.0172 - val_loss: 2.3838e-04 - val_mae: 0.0146
    Epoch 80/500
    43/43 [==============================] - 1s 18ms/step - loss: 9.8251e-05 - mae: 0.0111 - val_loss: 2.6916e-04 - val_mae: 0.0152
    Epoch 81/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.3113e-05 - mae: 0.0092 - val_loss: 3.0673e-04 - val_mae: 0.0167
    Epoch 82/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.9199e-05 - mae: 0.0087 - val_loss: 3.8825e-04 - val_mae: 0.0237
    Epoch 83/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.6651e-05 - mae: 0.0085 - val_loss: 5.3513e-04 - val_mae: 0.0255
    Epoch 84/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.7465e-05 - mae: 0.0085 - val_loss: 5.2859e-04 - val_mae: 0.0253
    Epoch 85/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.2883e-05 - mae: 0.0082 - val_loss: 3.5918e-04 - val_mae: 0.0190
    Epoch 86/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.9616e-05 - mae: 0.0085 - val_loss: 0.0013 - val_mae: 0.0468
    Epoch 87/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.4492e-05 - mae: 0.0084 - val_loss: 2.8320e-04 - val_mae: 0.0193
    Epoch 88/500
    43/43 [==============================] - 1s 21ms/step - loss: 8.9155e-05 - mae: 0.0093 - val_loss: 0.0035 - val_mae: 0.0802
    Epoch 89/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.8532e-04 - mae: 0.0126 - val_loss: 0.0169 - val_mae: 0.1806
    Epoch 90/500
    40/43 [==========================>...] - ETA: 0s - loss: 1.2953e-04 - mae: 0.0121INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 27ms/step - loss: 1.2638e-04 - mae: 0.0119 - val_loss: 2.1172e-04 - val_mae: 0.0148
    Epoch 91/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.5331e-04 - mae: 0.0111 - val_loss: 0.0121 - val_mae: 0.1526
    Epoch 92/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1104e-04 - mae: 0.0114 - val_loss: 9.9723e-04 - val_mae: 0.0395
    Epoch 93/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.4646e-04 - mae: 0.0112 - val_loss: 0.0102 - val_mae: 0.1399
    Epoch 94/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.6999e-04 - mae: 0.0152 - val_loss: 0.0213 - val_mae: 0.2029
    Epoch 95/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.9563e-04 - mae: 0.0148 - val_loss: 0.0056 - val_mae: 0.1034
    Epoch 96/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.2895e-04 - mae: 0.0116 - val_loss: 0.0074 - val_mae: 0.1190
    Epoch 97/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1521e-04 - mae: 0.0115 - val_loss: 0.0036 - val_mae: 0.0812
    Epoch 98/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.7786e-05 - mae: 0.0094 - val_loss: 0.0020 - val_mae: 0.0589
    Epoch 99/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.6625e-04 - mae: 0.0137 - val_loss: 0.0169 - val_mae: 0.1810
    Epoch 100/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1992e-04 - mae: 0.0117 - val_loss: 0.0013 - val_mae: 0.0457
    Epoch 101/500
    43/43 [==============================] - ETA: 0s - loss: 5.9928e-05 - mae: 0.0082INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 29ms/step - loss: 5.9928e-05 - mae: 0.0082 - val_loss: 2.0259e-04 - val_mae: 0.0148
    Epoch 102/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.7421e-05 - mae: 0.0079 - val_loss: 0.0011 - val_mae: 0.0416
    Epoch 103/500
    43/43 [==============================] - 1s 19ms/step - loss: 7.0938e-05 - mae: 0.0086 - val_loss: 0.0021 - val_mae: 0.0616
    Epoch 104/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.9503e-04 - mae: 0.0136 - val_loss: 0.0373 - val_mae: 0.2693
    Epoch 105/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3472e-04 - mae: 0.0158 - val_loss: 9.7447e-04 - val_mae: 0.0394
    Epoch 106/500
    37/43 [========================>.....] - ETA: 0s - loss: 6.3148e-05 - mae: 0.0083INFO:tensorflow:Assets written to: my_checkpoint/assets
    43/43 [==============================] - 1s 30ms/step - loss: 6.4445e-05 - mae: 0.0085 - val_loss: 1.9007e-04 - val_mae: 0.0137
    Epoch 107/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.9364e-04 - mae: 0.0130 - val_loss: 0.0414 - val_mae: 0.2836
    Epoch 108/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.6052e-04 - mae: 0.0166 - val_loss: 6.3941e-04 - val_mae: 0.0301
    Epoch 109/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.0048e-04 - mae: 0.0102 - val_loss: 0.0044 - val_mae: 0.0907
    Epoch 110/500
    43/43 [==============================] - 1s 20ms/step - loss: 7.5640e-05 - mae: 0.0094 - val_loss: 6.5307e-04 - val_mae: 0.0304
    Epoch 111/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.0837e-04 - mae: 0.0099 - val_loss: 0.0049 - val_mae: 0.0962
    Epoch 112/500
    43/43 [==============================] - 1s 18ms/step - loss: 8.0640e-05 - mae: 0.0096 - val_loss: 0.0016 - val_mae: 0.0535
    Epoch 113/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.4767e-04 - mae: 0.0111 - val_loss: 0.0172 - val_mae: 0.1824
    Epoch 114/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.5854e-04 - mae: 0.0135 - val_loss: 0.0055 - val_mae: 0.1021
    Epoch 115/500
    43/43 [==============================] - 1s 19ms/step - loss: 9.7914e-05 - mae: 0.0106 - val_loss: 0.0028 - val_mae: 0.0715
    Epoch 116/500
    43/43 [==============================] - 1s 18ms/step - loss: 7.2016e-05 - mae: 0.0090 - val_loss: 0.0019 - val_mae: 0.0592
    Epoch 117/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.9175e-05 - mae: 0.0082 - val_loss: 5.7452e-04 - val_mae: 0.0281
    Epoch 118/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.7664e-04 - mae: 0.0140 - val_loss: 0.0327 - val_mae: 0.2525
    Epoch 119/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5273e-04 - mae: 0.0187 - val_loss: 0.0069 - val_mae: 0.1149
    Epoch 120/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1029e-04 - mae: 0.0114 - val_loss: 0.0011 - val_mae: 0.0424
    Epoch 121/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.5445e-05 - mae: 0.0085 - val_loss: 0.0012 - val_mae: 0.0437
    Epoch 122/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.4603e-05 - mae: 0.0078 - val_loss: 2.3933e-04 - val_mae: 0.0177
    Epoch 123/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.5345e-05 - mae: 0.0091 - val_loss: 0.0031 - val_mae: 0.0761
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
