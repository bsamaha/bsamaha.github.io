---
title: "Project: Part 6, Time Series LSTM Forecast with Keras"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Deep Learning
  - Keras
  - CNN
  - RNN
  - TensorFlow
---
**This notebook is part 6 of a multi-phase project of building a quantitative trading algorithm. This notebook is about adding a CNN preprocessing layer with RNN layers models to see if it provides better predictive power.**


# Partial, and Full CNN predictiong S&P500 Daily Closing Price

The first model we will build will have a 1D CNN layer that acts as a preproccesser of our SPY data. A 1D-Conv layer acts exactly the same as a 2-D layer used in image processing except that it only operates in one dimension. In image processing the dimensions are height and width, while in our example here we are only worried 1 dimension-- Closing price.

As discussed previously, RNNs have memory stored in their hidden states. A 1D CNN layer does not have any memory as it only calculates an output based on the current filter size. A filter/kernel is just like our moving window.

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
  

def seq2seq_window_dataset(series, window_size, batch_size=128,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
  
def sequential_window_dataset(series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
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
    


![png](/assets/images/Preprocess_CNN/output_5_1.png)



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

## Preprocessing With 1D-Convolutional Layers

### Padding

*   No padding - does not add 0's in place of missing data of input sequence. This shortens the sequence length due to the fact that the output will be missing values at the beginning and end due to window size.
*   Same padding - Same padding adds zeros to the left and right when data is missing in the kernel so that the output size matches the input size.
*   Causal Padding -  instead of padding 0's to the left and right it only pads 0's to the left. This is important when building a model for forecasting ensuring the model does not cheat and use future values to predict future values.


### Stride
Stride is how many time steps the kernel/window moves after calculating an input. For a stride of 1 and a kernel size of 3 it will first calculate an output on steps [1,2,3] and then move 1 stride to [2,3,4].


### Filters
The number of filters means how many different kernels are tested on each input sequence. A number too low the model will perform poorly while too many will overfit the data.


```python
# Clear any backend results so that we start fresh
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Set window size and create input sequence batches
window_size = 20
train_set = seq2seq_window_dataset(normalized_x_train, window_size,
                                   batch_size=128)

# Create model
model = keras.models.Sequential([
  keras.layers.Conv1D(filters=32, kernel_size=10,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  keras.layers.LSTM(32, return_sequences=True),
  keras.layers.LSTM(32, return_sequences=True),
  keras.layers.Dense(1),
])

# Set learning rate finder
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-5 * 10**(epoch / 20))

# Choose optimizer
optimizer = keras.optimizers.Nadam(lr=1e-5)

# Compile Model
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Save history to plot learning rate
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

    Epoch 1/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0921 - mae: 0.3706
    Epoch 2/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0879 - mae: 0.3609
    Epoch 3/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0839 - mae: 0.3514
    Epoch 4/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0794 - mae: 0.3411
    Epoch 5/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0745 - mae: 0.3295
    Epoch 6/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0689 - mae: 0.3160
    Epoch 7/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0626 - mae: 0.2997
    Epoch 8/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0555 - mae: 0.2798
    Epoch 9/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0475 - mae: 0.2552
    Epoch 10/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0394 - mae: 0.2254
    Epoch 11/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0325 - mae: 0.1926
    Epoch 12/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0283 - mae: 0.1720
    Epoch 13/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0265 - mae: 0.1692
    Epoch 14/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0255 - mae: 0.1673
    Epoch 15/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0244 - mae: 0.1634
    Epoch 16/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0232 - mae: 0.1582
    Epoch 17/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0218 - mae: 0.1516
    Epoch 18/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0204 - mae: 0.1448
    Epoch 19/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0187 - mae: 0.1363
    Epoch 20/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0169 - mae: 0.1258
    Epoch 21/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0151 - mae: 0.1152
    Epoch 22/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0135 - mae: 0.1064
    Epoch 23/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0121 - mae: 0.1001
    Epoch 24/100
    43/43 [==============================] - 0s 12ms/step - loss: 0.0109 - mae: 0.0929
    Epoch 25/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0097 - mae: 0.0847
    Epoch 26/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0086 - mae: 0.0759
    Epoch 27/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0076 - mae: 0.0678
    Epoch 28/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0068 - mae: 0.0622
    Epoch 29/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0062 - mae: 0.0589
    Epoch 30/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0056 - mae: 0.0557
    Epoch 31/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0051 - mae: 0.0521
    Epoch 32/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0046 - mae: 0.0495
    Epoch 33/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0041 - mae: 0.0463
    Epoch 34/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0037 - mae: 0.0449
    Epoch 35/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0031 - mae: 0.0413
    Epoch 36/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0029 - mae: 0.0408
    Epoch 37/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0023 - mae: 0.0371
    Epoch 38/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0017 - mae: 0.0311
    Epoch 39/100
    43/43 [==============================] - 1s 18ms/step - loss: 0.0015 - mae: 0.0317
    Epoch 40/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0026 - mae: 0.0428
    Epoch 41/100
    43/43 [==============================] - 1s 13ms/step - loss: 7.0207e-04 - mae: 0.0223
    Epoch 42/100
    43/43 [==============================] - 1s 13ms/step - loss: 4.3753e-04 - mae: 0.0185
    Epoch 43/100
    43/43 [==============================] - 1s 17ms/step - loss: 2.9257e-04 - mae: 0.0159
    Epoch 44/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0025 - mae: 0.0312
    Epoch 45/100
    43/43 [==============================] - 1s 16ms/step - loss: 3.2363e-04 - mae: 0.0189
    Epoch 46/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.8532e-04 - mae: 0.0140
    Epoch 47/100
    43/43 [==============================] - 1s 17ms/step - loss: 4.5749e-04 - mae: 0.0177
    Epoch 48/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0021 - mae: 0.0394
    Epoch 49/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.8676e-04 - mae: 0.0146
    Epoch 50/100
    43/43 [==============================] - 1s 13ms/step - loss: 4.0938e-04 - mae: 0.0149
    Epoch 51/100
    43/43 [==============================] - 1s 14ms/step - loss: 7.9299e-04 - mae: 0.0215
    Epoch 52/100
    43/43 [==============================] - 1s 16ms/step - loss: 3.8903e-04 - mae: 0.0184
    Epoch 53/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.7924e-04 - mae: 0.0135
    Epoch 54/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0016 - mae: 0.0225
    Epoch 55/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0014 - mae: 0.0275
    Epoch 56/100
    43/43 [==============================] - 1s 15ms/step - loss: 9.0794e-04 - mae: 0.0239
    Epoch 57/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0011 - mae: 0.0236
    Epoch 58/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.7238e-04 - mae: 0.0157
    Epoch 59/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0018 - mae: 0.0289
    Epoch 60/100
    43/43 [==============================] - 1s 13ms/step - loss: 2.4012e-04 - mae: 0.0146
    Epoch 61/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0036 - mae: 0.0373
    Epoch 62/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0024 - mae: 0.0355
    Epoch 63/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0014 - mae: 0.0275
    Epoch 64/100
    43/43 [==============================] - 1s 13ms/step - loss: 7.0109e-04 - mae: 0.0193
    Epoch 65/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0020 - mae: 0.0362
    Epoch 66/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0026 - mae: 0.0391
    Epoch 67/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0020 - mae: 0.0309
    Epoch 68/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0020 - mae: 0.0311
    Epoch 69/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0032 - mae: 0.0385
    Epoch 70/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0023 - mae: 0.0349
    Epoch 71/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0011 - mae: 0.0209
    Epoch 72/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.0029 - mae: 0.0387
    Epoch 73/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0382 - mae: 0.1214
    Epoch 74/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.9550 - mae: 1.3977
    Epoch 75/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.0900 - mae: 0.3333
    Epoch 76/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0651 - mae: 0.2876
    Epoch 77/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.1485 - mae: 0.4464
    Epoch 78/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1567 - mae: 0.4703
    Epoch 79/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.1179 - mae: 0.4141
    Epoch 80/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1828 - mae: 0.5488
    Epoch 81/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.2360 - mae: 0.6277
    Epoch 82/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.3112 - mae: 0.7452
    Epoch 83/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.4000 - mae: 0.8539
    Epoch 84/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.3843 - mae: 0.8226
    Epoch 85/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.7205 - mae: 1.2126
    Epoch 86/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.6103 - mae: 1.0986
    Epoch 87/100
    43/43 [==============================] - 1s 12ms/step - loss: 0.6755 - mae: 1.1715
    Epoch 88/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.6995 - mae: 1.1910
    Epoch 89/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.7602 - mae: 1.2468
    Epoch 90/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.0975 - mae: 1.5933
    Epoch 91/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.9910 - mae: 1.4889
    Epoch 92/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.8755 - mae: 1.3718
    Epoch 93/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.9127 - mae: 1.4107
    Epoch 94/100
    43/43 [==============================] - 1s 13ms/step - loss: 0.8724 - mae: 1.3690
    Epoch 95/100
    43/43 [==============================] - 1s 13ms/step - loss: 1.2592 - mae: 1.7549
    Epoch 96/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.7311 - mae: 1.2272
    Epoch 97/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.7122 - mae: 1.2110
    Epoch 98/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.6995 - mae: 1.1956
    Epoch 99/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.7937 - mae: 1.2933
    Epoch 100/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.8416 - mae: 1.3392
    


```python
# Plot learning rate
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1, 0, .1])
```




    (1e-06, 1.0, 0.0, 0.1)




![png](/assets/images/Preprocess_CNN/output_10_1.png)



```python
# Clear any backend results so that we start fresh

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Set window size and create input sequence batches
window_size = 20
train_set = seq2seq_window_dataset(normalized_x_train, window_size,
                                   batch_size=128)
valid_set = seq2seq_window_dataset(normalized_x_valid, window_size,
                                   batch_size=128)

# Create model
model = keras.models.Sequential([
  keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  keras.layers.LSTM(32, return_sequences=True),
  keras.layers.LSTM(32, return_sequences=True),
  keras.layers.Dense(1),
])

# Choose optimizer
optimizer = keras.optimizers.Nadam(lr=1e-3)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Create recall for best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint.h5", save_best_only=True)

# Set early stopping
early_stopping = keras.callbacks.EarlyStopping(patience=50)

# Fit model to data
model.fit(train_set, epochs=500,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint])
```

    Epoch 1/500
    43/43 [==============================] - 1s 33ms/step - loss: 0.0126 - mae: 0.1116 - val_loss: 0.0735 - val_mae: 0.2485
    Epoch 2/500
    43/43 [==============================] - 1s 16ms/step - loss: 0.0066 - mae: 0.0691 - val_loss: 0.0471 - val_mae: 0.1728
    Epoch 3/500
    43/43 [==============================] - 1s 17ms/step - loss: 0.0039 - mae: 0.0452 - val_loss: 0.0357 - val_mae: 0.1189
    Epoch 4/500
    43/43 [==============================] - 1s 14ms/step - loss: 0.0030 - mae: 0.0398 - val_loss: 0.0331 - val_mae: 0.1601
    Epoch 5/500
    43/43 [==============================] - 1s 17ms/step - loss: 0.0030 - mae: 0.0428 - val_loss: 0.0223 - val_mae: 0.0959
    Epoch 6/500
    43/43 [==============================] - 1s 18ms/step - loss: 0.0018 - mae: 0.0302 - val_loss: 0.0188 - val_mae: 0.0945
    Epoch 7/500
    43/43 [==============================] - 1s 18ms/step - loss: 0.0014 - mae: 0.0251 - val_loss: 0.0146 - val_mae: 0.0820
    Epoch 8/500
    43/43 [==============================] - 1s 17ms/step - loss: 0.0011 - mae: 0.0211 - val_loss: 0.0116 - val_mae: 0.0759
    Epoch 9/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.5730e-04 - mae: 0.0202 - val_loss: 0.0092 - val_mae: 0.0790
    Epoch 10/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.9424e-04 - mae: 0.0197 - val_loss: 0.0083 - val_mae: 0.0928
    Epoch 11/500
    43/43 [==============================] - 1s 20ms/step - loss: 5.0770e-04 - mae: 0.0165 - val_loss: 0.0045 - val_mae: 0.0397
    Epoch 12/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.8892e-04 - mae: 0.0149 - val_loss: 0.0044 - val_mae: 0.0721
    Epoch 13/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.3226e-04 - mae: 0.0186 - val_loss: 0.0332 - val_mae: 0.2469
    Epoch 14/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.9635e-04 - mae: 0.0164 - val_loss: 0.0038 - val_mae: 0.0748
    Epoch 15/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.9545e-04 - mae: 0.0127 - val_loss: 0.0055 - val_mae: 0.0947
    Epoch 16/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.2065e-04 - mae: 0.0100 - val_loss: 0.0014 - val_mae: 0.0433
    Epoch 17/500
    43/43 [==============================] - 1s 18ms/step - loss: 9.0288e-05 - mae: 0.0086 - val_loss: 7.6500e-04 - val_mae: 0.0294
    Epoch 18/500
    43/43 [==============================] - 1s 15ms/step - loss: 9.6842e-05 - mae: 0.0090 - val_loss: 0.0057 - val_mae: 0.0964
    Epoch 19/500
    43/43 [==============================] - 1s 16ms/step - loss: 9.9874e-05 - mae: 0.0099 - val_loss: 0.0089 - val_mae: 0.1229
    Epoch 20/500
    43/43 [==============================] - 1s 16ms/step - loss: 7.6657e-04 - mae: 0.0206 - val_loss: 0.0016 - val_mae: 0.0420
    Epoch 21/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.9347e-05 - mae: 0.0089 - val_loss: 0.0024 - val_mae: 0.0554
    Epoch 22/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.2290e-05 - mae: 0.0069 - val_loss: 0.0017 - val_mae: 0.0450
    Epoch 23/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.6395e-05 - mae: 0.0065 - val_loss: 0.0015 - val_mae: 0.0429
    Epoch 24/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.4899e-05 - mae: 0.0065 - val_loss: 0.0016 - val_mae: 0.0443
    Epoch 25/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.2317e-05 - mae: 0.0063 - val_loss: 0.0014 - val_mae: 0.0404
    Epoch 26/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.1030e-05 - mae: 0.0062 - val_loss: 0.0016 - val_mae: 0.0452
    Epoch 27/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9756e-05 - mae: 0.0061 - val_loss: 0.0015 - val_mae: 0.0434
    Epoch 28/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.8117e-05 - mae: 0.0060 - val_loss: 0.0012 - val_mae: 0.0378
    Epoch 29/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.8070e-05 - mae: 0.0060 - val_loss: 7.7088e-04 - val_mae: 0.0299
    Epoch 30/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.7762e-05 - mae: 0.0060 - val_loss: 0.0020 - val_mae: 0.0518
    Epoch 31/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5782e-05 - mae: 0.0059 - val_loss: 9.8866e-04 - val_mae: 0.0337
    Epoch 32/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.2056e-05 - mae: 0.0063 - val_loss: 0.0034 - val_mae: 0.0719
    Epoch 33/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.7262e-05 - mae: 0.0062 - val_loss: 6.9099e-04 - val_mae: 0.0289
    Epoch 34/500
    43/43 [==============================] - 1s 22ms/step - loss: 5.2978e-05 - mae: 0.0068 - val_loss: 0.0082 - val_mae: 0.1176
    Epoch 35/500
    43/43 [==============================] - 1s 17ms/step - loss: 9.3577e-05 - mae: 0.0091 - val_loss: 0.0019 - val_mae: 0.0543
    Epoch 36/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.8048e-04 - mae: 0.0115 - val_loss: 0.0273 - val_mae: 0.2208
    Epoch 37/500
    43/43 [==============================] - 1s 16ms/step - loss: 1.0419e-04 - mae: 0.0103 - val_loss: 0.0043 - val_mae: 0.0791
    Epoch 38/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.9595e-05 - mae: 0.0067 - val_loss: 9.3054e-04 - val_mae: 0.0324
    Epoch 39/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.0443e-05 - mae: 0.0073 - val_loss: 0.0088 - val_mae: 0.1210
    Epoch 40/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.1570e-05 - mae: 0.0067 - val_loss: 0.0017 - val_mae: 0.0440
    Epoch 41/500
    43/43 [==============================] - 1s 17ms/step - loss: 4.0895e-05 - mae: 0.0063 - val_loss: 0.0059 - val_mae: 0.0963
    Epoch 42/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.3731e-05 - mae: 0.0060 - val_loss: 0.0019 - val_mae: 0.0478
    Epoch 43/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.8066e-05 - mae: 0.0065 - val_loss: 0.0012 - val_mae: 0.0418
    Epoch 44/500
    43/43 [==============================] - 1s 15ms/step - loss: 1.9006e-04 - mae: 0.0107 - val_loss: 0.0037 - val_mae: 0.0797
    Epoch 45/500
    43/43 [==============================] - 1s 17ms/step - loss: 7.2978e-05 - mae: 0.0086 - val_loss: 9.3000e-04 - val_mae: 0.0317
    Epoch 46/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.9451e-05 - mae: 0.0072 - val_loss: 0.0017 - val_mae: 0.0504
    Epoch 47/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.7883e-05 - mae: 0.0073 - val_loss: 5.9581e-04 - val_mae: 0.0292
    Epoch 48/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.1365e-04 - mae: 0.0090 - val_loss: 0.0189 - val_mae: 0.1839
    Epoch 49/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1302e-04 - mae: 0.0106 - val_loss: 0.0092 - val_mae: 0.1244
    Epoch 50/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.5740e-05 - mae: 0.0069 - val_loss: 0.0013 - val_mae: 0.0364
    Epoch 51/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.1992e-05 - mae: 0.0063 - val_loss: 7.8528e-04 - val_mae: 0.0342
    Epoch 52/500
    43/43 [==============================] - 1s 19ms/step - loss: 8.4255e-05 - mae: 0.0085 - val_loss: 0.0110 - val_mae: 0.1376
    Epoch 53/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.8356e-05 - mae: 0.0071 - val_loss: 0.0025 - val_mae: 0.0553
    Epoch 54/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.9266e-05 - mae: 0.0054 - val_loss: 0.0038 - val_mae: 0.0738
    Epoch 55/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.7332e-05 - mae: 0.0054 - val_loss: 0.0011 - val_mae: 0.0335
    Epoch 56/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.5303e-05 - mae: 0.0074 - val_loss: 0.0033 - val_mae: 0.0753
    Epoch 57/500
    43/43 [==============================] - 1s 17ms/step - loss: 8.7286e-05 - mae: 0.0094 - val_loss: 0.0096 - val_mae: 0.1279
    Epoch 58/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.4948e-05 - mae: 0.0076 - val_loss: 0.0055 - val_mae: 0.0923
    Epoch 59/500
    43/43 [==============================] - 1s 16ms/step - loss: 8.1564e-05 - mae: 0.0081 - val_loss: 0.0137 - val_mae: 0.1548
    Epoch 60/500
    43/43 [==============================] - 1s 16ms/step - loss: 5.3821e-05 - mae: 0.0075 - val_loss: 0.0019 - val_mae: 0.0462
    Epoch 61/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.2137e-05 - mae: 0.0062 - val_loss: 0.0066 - val_mae: 0.1029
    Epoch 62/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.0113e-05 - mae: 0.0065 - val_loss: 0.0066 - val_mae: 0.1026
    Epoch 63/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.6766e-05 - mae: 0.0072 - val_loss: 0.0098 - val_mae: 0.1291
    Epoch 64/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.4182e-05 - mae: 0.0080 - val_loss: 0.0095 - val_mae: 0.1260
    Epoch 65/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.0287e-04 - mae: 0.0092 - val_loss: 0.0156 - val_mae: 0.1652
    Epoch 66/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.2392e-04 - mae: 0.0106 - val_loss: 0.0015 - val_mae: 0.0480
    Epoch 67/500
    43/43 [==============================] - 1s 17ms/step - loss: 9.1070e-05 - mae: 0.0092 - val_loss: 8.8045e-04 - val_mae: 0.0365
    Epoch 68/500
    43/43 [==============================] - 1s 16ms/step - loss: 4.0833e-05 - mae: 0.0068 - val_loss: 0.0012 - val_mae: 0.0351
    Epoch 69/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.2568e-05 - mae: 0.0058 - val_loss: 8.0016e-04 - val_mae: 0.0304
    Epoch 70/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.5413e-05 - mae: 0.0052 - val_loss: 0.0016 - val_mae: 0.0414
    Epoch 71/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.4774e-05 - mae: 0.0050 - val_loss: 0.0028 - val_mae: 0.0619
    Epoch 72/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.9709e-05 - mae: 0.0054 - val_loss: 6.6531e-04 - val_mae: 0.0301
    Epoch 73/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.4780e-05 - mae: 0.0072 - val_loss: 0.0142 - val_mae: 0.1585
    Epoch 74/500
    43/43 [==============================] - 1s 15ms/step - loss: 1.2875e-04 - mae: 0.0105 - val_loss: 0.0021 - val_mae: 0.0568
    Epoch 75/500
    43/43 [==============================] - 1s 16ms/step - loss: 1.1458e-04 - mae: 0.0100 - val_loss: 0.0129 - val_mae: 0.1498
    Epoch 76/500
    43/43 [==============================] - 1s 17ms/step - loss: 5.9714e-05 - mae: 0.0079 - val_loss: 0.0012 - val_mae: 0.0358
    Epoch 77/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.6031e-05 - mae: 0.0053 - val_loss: 0.0025 - val_mae: 0.0564
    Epoch 78/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.3850e-05 - mae: 0.0049 - val_loss: 0.0012 - val_mae: 0.0355
    Epoch 79/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.6391e-05 - mae: 0.0051 - val_loss: 0.0036 - val_mae: 0.0714
    Epoch 80/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.8329e-05 - mae: 0.0053 - val_loss: 0.0042 - val_mae: 0.0789
    Epoch 81/500
    43/43 [==============================] - 1s 17ms/step - loss: 1.0305e-04 - mae: 0.0083 - val_loss: 0.0186 - val_mae: 0.1820
    Epoch 82/500
    43/43 [==============================] - 1s 15ms/step - loss: 7.1615e-05 - mae: 0.0086 - val_loss: 0.0019 - val_mae: 0.0441
    Epoch 83/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.2673e-05 - mae: 0.0057 - val_loss: 0.0013 - val_mae: 0.0366
    Epoch 84/500
    43/43 [==============================] - 1s 16ms/step - loss: 3.9983e-05 - mae: 0.0059 - val_loss: 0.0082 - val_mae: 0.1172
    Epoch 85/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.5447e-04 - mae: 0.0114 - val_loss: 0.0456 - val_mae: 0.2906
    Epoch 86/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.1867e-04 - mae: 0.0144 - val_loss: 0.0046 - val_mae: 0.0757
    Epoch 87/500
    43/43 [==============================] - 1s 16ms/step - loss: 6.2281e-05 - mae: 0.0070 - val_loss: 0.0020 - val_mae: 0.0442
    Epoch 88/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.1997e-05 - mae: 0.0073 - val_loss: 9.7070e-04 - val_mae: 0.0388
    Epoch 89/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.5394e-05 - mae: 0.0061 - val_loss: 0.0022 - val_mae: 0.0501
    Epoch 90/500
    43/43 [==============================] - 1s 17ms/step - loss: 3.0410e-05 - mae: 0.0054 - val_loss: 0.0053 - val_mae: 0.0908
    Epoch 91/500
    43/43 [==============================] - 1s 17ms/step - loss: 2.8890e-05 - mae: 0.0056 - val_loss: 0.0034 - val_mae: 0.0673
    Epoch 92/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.2646e-05 - mae: 0.0048 - val_loss: 0.0024 - val_mae: 0.0533
    Epoch 93/500
    43/43 [==============================] - 1s 17ms/step - loss: 6.0053e-05 - mae: 0.0065 - val_loss: 0.0133 - val_mae: 0.1524
    Epoch 94/500
    43/43 [==============================] - 1s 20ms/step - loss: 7.1070e-05 - mae: 0.0086 - val_loss: 0.0082 - val_mae: 0.1149
    Epoch 95/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.3829e-05 - mae: 0.0061 - val_loss: 0.0040 - val_mae: 0.0732
    Epoch 96/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.4640e-05 - mae: 0.0057 - val_loss: 0.0062 - val_mae: 0.0977
    Epoch 97/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.2799e-05 - mae: 0.0065 - val_loss: 0.0073 - val_mae: 0.1085
    




    <tensorflow.python.keras.callbacks.History at 0x7faeb0708048>




```python
# recall best model
model = keras.models.load_model("my_checkpoint.h5")
```


```python
# Forecast test data
rnn_forecast = model_forecast(model, spy_normalized_to_traindata[:,  np.newaxis], window_size)
rnn_forecast = rnn_forecast[x_test.index.min() - window_size:-1,-1,0]
```


```python
# Scale data back to normal
rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()
rnn_unscaled_forecast.shape
```




    (422,)




```python
# Plot model results vs actual
plt.figure(figsize=(10, 6))

plt.ylabel('Dollars $')
plt.xlabel('Timestep in Days')
plt.title(f'Preprocess with CNN Forecast {window_size}')
plot_series(x_test.index, x_test, label = 'Actual')
plot_series(x_test.index, rnn_unscaled_forecast, label = 'Forecast')
```


![png](/assets/images/Preprocess_CNN/output_15_0.png)



```python
# Calculate MAE
keras.metrics.mean_absolute_error(x_test, rnn_unscaled_forecast).numpy()
```




    12.655453




```python

```
