---
title: "Project: Part 4 RNN Forecast with Keras"
categories:
  - Project
  - Finance
tags:
  - Time Series Analysis
  - Deep Learning
  - Keras
  - RNN
  - Tensorflow
---
**This notebook is part 4 of a multi-phase project of building a quantitative trading algorithm. This notebook is about RNNs and shows how using a sequence to sequence RNN helps create a faster training model.**

# What is a Recurrent Neural Network?

A recurrent layer is just a memory cell that computes. In diagrams, you may see it represented as having many cells, but it is just one cell used to calculate an output over a set amount of time steps. The cell calculates Y_0 at X_0 then moves on to calculating Y_1 at X_1 and so on. However, from X_0 to X_1 the memory cell produces a state vector. This state factor is used in the next time step as an additional input factor. This state factor is why it is called a recurrent neural network. Just as x +=1 is a recurrent function to add 1 every time to X, every time step receives an state vector from the last time step.

This architecture enables you to use any sequence length as long as the parameters remain constant. An RNN can be though of as word context of reading a sentence. As you read the a sentence you may only be focusing on one word, however, if you encounter a word that is a homphone such as lead you will understand the context. You can immediate notice if lead is the verb for "guiding" or the soft metal which is a noun.

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


```python
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
  
  
def window_dataset(series, window_size, batch_size=128,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
  
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
spy = pd.read_csv('../SPY.csv')

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

    5521 5522 6527 6528 6954
    


![png](/assets/images/RNN_seqtovec_seqtoseq/output_5_1.png)


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

## Simple RNN Forecasting

The input for the memory cell at each time step is the batch size by our feature dimensionality (1). The output is thiese same two dimensions times the number of units in the memory cell. Our memory cell is comprised of 100 units in both layers. So the output of our RNN layer is batch(128), window_size(30), and number of units (100) which is obviously 3 dimensional. The output Y_0 is the state vector which is used when calculating Y_1 at the next time step. 

In this instance we are doing sequence to vector which means we ignore all outputs except for the one at the very last time step. This is the default behavior of all reccurent layers in Keras unless return_sequences = True is selected. This sequence to vector takes in a batch (128) of time windows and outputs the next time step of window of values. This one at a time output proves to be very slow when training.

For a faster training convergence, we use a sequence to sequence RNN. Compared to the sequence to vector which adjusted the gradient of the loss from the very end of the model all the at layer 2 unit 100, the sequence to sequence RNN calculates the loss at each time step and backpropagates the loss from there. This provides much more gradients and speeds up training. It is important to note we still ignore all outputs besides the last one. We just calculate all intermediate values to update the gradient more quickly.


```python

```




```python
# Clear any back end stored data due to multiple iterations
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Set window size
window_size = 30

# Create 2D batches of batch size and features (1 feature = 1 time step in window)
train_set = window_dataset(normalized_x_train, window_size, batch_size=128)

# Establish Model
model = keras.models.Sequential([
  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), # Add a 3rd dimension (feature dimensionality which in this case in univariate)
                      input_shape=[None]), # 1st dimension is batch size, input shape = None allows windows of any size to be used
  keras.layers.SimpleRNN(100, return_sequences=True), # takes in sequence (batch size, time, dimensionality per time step (univariate))
  keras.layers.SimpleRNN(100), # produces a single vector
  keras.layers.Dense(1), # produces 1 output 
])

# create standard learning rate scheduler
lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-5 * 10**(epoch / 20))

# establish optimizer
optimizer = keras.optimizers.Nadam(lr=1e-7)

# Put model all together
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

    Epoch 1/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0200 - mae: 0.1493
    Epoch 2/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0130 - mae: 0.1190
    Epoch 3/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0052 - mae: 0.0707
    Epoch 4/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0023 - mae: 0.0452
    Epoch 5/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0012 - mae: 0.0321
    Epoch 6/100
    43/43 [==============================] - 1s 15ms/step - loss: 6.5596e-04 - mae: 0.0238
    Epoch 7/100
    43/43 [==============================] - 1s 15ms/step - loss: 4.4538e-04 - mae: 0.0196
    Epoch 8/100
    43/43 [==============================] - 1s 15ms/step - loss: 3.4397e-04 - mae: 0.0180
    Epoch 9/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.8254e-04 - mae: 0.0169
    Epoch 10/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.4019e-04 - mae: 0.0158
    Epoch 11/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.2150e-04 - mae: 0.0154
    Epoch 12/100
    43/43 [==============================] - 1s 15ms/step - loss: 2.2434e-04 - mae: 0.0158: 0s - loss: 1.8478e-04 - mae: 0.01
    Epoch 13/100
    43/43 [==============================] - 1s 15ms/step - loss: 6.1106e-04 - mae: 0.0245
    Epoch 14/100
    43/43 [==============================] - 1s 15ms/step - loss: 3.9755e-04 - mae: 0.0212
    Epoch 15/100
    43/43 [==============================] - 1s 15ms/step - loss: 4.9605e-04 - mae: 0.0209
    Epoch 16/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.5898e-04 - mae: 0.0133
    Epoch 17/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0018 - mae: 0.0385
    Epoch 18/100
    43/43 [==============================] - 1s 15ms/step - loss: 9.5971e-05 - mae: 0.0102
    Epoch 19/100
    43/43 [==============================] - 1s 15ms/step - loss: 5.1704e-04 - mae: 0.0195
    Epoch 20/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0014 - mae: 0.0317
    Epoch 21/100
    43/43 [==============================] - 1s 15ms/step - loss: 2.3174e-04 - mae: 0.0162
    Epoch 22/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0029 - mae: 0.0459
    Epoch 23/100
    43/43 [==============================] - 1s 15ms/step - loss: 6.7260e-05 - mae: 0.0084
    Epoch 24/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0013 - mae: 0.0307
    Epoch 25/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0016 - mae: 0.0466
    Epoch 26/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.3833e-04 - mae: 0.0124
    Epoch 27/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0039 - mae: 0.0579
    Epoch 28/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.7711e-04 - mae: 0.0128
    Epoch 29/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0030 - mae: 0.0438
    Epoch 30/100
    43/43 [==============================] - 1s 15ms/step - loss: 6.7489e-04 - mae: 0.0289
    Epoch 31/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0022 - mae: 0.0415
    Epoch 32/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0017 - mae: 0.0457
    Epoch 33/100
    43/43 [==============================] - 1s 15ms/step - loss: 7.9416e-05 - mae: 0.0096
    Epoch 34/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0732 - mae: 0.1863
    Epoch 35/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0701 - mae: 0.2809
    Epoch 36/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0115 - mae: 0.1120
    Epoch 37/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0056 - mae: 0.0631
    Epoch 38/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0077 - mae: 0.0872
    Epoch 39/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2479 - mae: 0.5360
    Epoch 40/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0179 - mae: 0.1372
    Epoch 41/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0450 - mae: 0.1857
    Epoch 42/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0147 - mae: 0.1281
    Epoch 43/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0139 - mae: 0.1260
    Epoch 44/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0132 - mae: 0.1249
    Epoch 45/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0142 - mae: 0.1283
    Epoch 46/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0134 - mae: 0.1259
    Epoch 47/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0136 - mae: 0.1261
    Epoch 48/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0137 - mae: 0.1266
    Epoch 49/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0142 - mae: 0.1279
    Epoch 50/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0149 - mae: 0.1320
    Epoch 51/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0773 - mae: 0.2248
    Epoch 52/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0120 - mae: 0.1235
    Epoch 53/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0187 - mae: 0.1477
    Epoch 54/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0182 - mae: 0.1468
    Epoch 55/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0282 - mae: 0.1741
    Epoch 56/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.0395 - mae: 0.2078
    Epoch 57/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0653 - mae: 0.2731
    Epoch 58/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.0495 - mae: 0.2456
    Epoch 59/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.1138 - mae: 0.3769
    Epoch 60/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.1084 - mae: 0.3555
    Epoch 61/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.1184 - mae: 0.4274: 0s - loss: 0.1914 - mae
    Epoch 62/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.2676 - mae: 0.6625
    Epoch 63/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.1760 - mae: 0.5215
    Epoch 64/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.3159 - mae: 0.7539
    Epoch 65/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.3103 - mae: 0.7468
    Epoch 66/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.3553 - mae: 0.7974
    Epoch 67/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.3363 - mae: 0.7187
    Epoch 68/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.5587 - mae: 1.0523
    Epoch 69/100
    43/43 [==============================] - 1s 14ms/step - loss: 0.5627 - mae: 1.0459
    Epoch 70/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.6706 - mae: 1.1693
    Epoch 71/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.7058 - mae: 1.2014
    Epoch 72/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.8149 - mae: 1.3146
    Epoch 73/100
    43/43 [==============================] - 1s 15ms/step - loss: 0.8807 - mae: 1.3789
    Epoch 74/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.0058 - mae: 1.5058
    Epoch 75/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.1098 - mae: 1.6097
    Epoch 76/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.2480 - mae: 1.7480
    Epoch 77/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.3964 - mae: 1.8964
    Epoch 78/100
    43/43 [==============================] - 1s 14ms/step - loss: 1.5633 - mae: 2.0633
    Epoch 79/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.7543 - mae: 2.2543
    Epoch 80/100
    43/43 [==============================] - 1s 15ms/step - loss: 1.9520 - mae: 2.4520
    Epoch 81/100
    43/43 [==============================] - 1s 14ms/step - loss: 2.2057 - mae: 2.7057
    Epoch 82/100
    43/43 [==============================] - 1s 14ms/step - loss: 2.4363 - mae: 2.9363
    Epoch 83/100
    43/43 [==============================] - 1s 15ms/step - loss: 2.7492 - mae: 3.2492
    Epoch 84/100
    43/43 [==============================] - 1s 15ms/step - loss: 3.0525 - mae: 3.5525
    Epoch 85/100
    43/43 [==============================] - 1s 15ms/step - loss: 3.4375 - mae: 3.9375
    Epoch 86/100
    43/43 [==============================] - 1s 14ms/step - loss: 3.8105 - mae: 4.3105
    Epoch 87/100
    43/43 [==============================] - 1s 15ms/step - loss: 4.2849 - mae: 4.7849
    Epoch 88/100
    43/43 [==============================] - 1s 15ms/step - loss: 4.7519 - mae: 5.2519
    Epoch 89/100
    43/43 [==============================] - 1s 15ms/step - loss: 5.3258 - mae: 5.8258
    Epoch 90/100
    43/43 [==============================] - 1s 15ms/step - loss: 5.9269 - mae: 6.4269
    Epoch 91/100
    43/43 [==============================] - 1s 15ms/step - loss: 6.6235 - mae: 7.1235
    Epoch 92/100
    43/43 [==============================] - 1s 16ms/step - loss: 7.3712 - mae: 7.8712
    Epoch 93/100
    43/43 [==============================] - 1s 16ms/step - loss: 8.2384 - mae: 8.7384
    Epoch 94/100
    43/43 [==============================] - 1s 15ms/step - loss: 9.1723 - mae: 9.6723
    Epoch 95/100
    43/43 [==============================] - 1s 15ms/step - loss: 10.2099 - mae: 10.7099
    Epoch 96/100
    43/43 [==============================] - 1s 15ms/step - loss: 11.3704 - mae: 11.8704
    Epoch 97/100
    43/43 [==============================] - 1s 15ms/step - loss: 12.6061 - mae: 13.1061
    Epoch 98/100
    43/43 [==============================] - 1s 15ms/step - loss: 14.2846 - mae: 14.7846
    Epoch 99/100
    43/43 [==============================] - 1s 15ms/step - loss: 15.6790 - mae: 16.1790
    Epoch 100/100
    43/43 [==============================] - 1s 15ms/step - loss: 17.7097 - mae: 18.2097
    


```python
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-5, 1, 0, .1])
```




    (1e-05, 1.0, 0.0, 0.1)




![png](/assets/images/RNN_seqtovec_seqtoseq/output_12_1.png)



```python
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = window_dataset(normalized_x_train, window_size, batch_size=128)
valid_set = window_dataset(normalized_x_valid, window_size, batch_size=128)

model = keras.models.Sequential([
  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.SimpleRNN(100),
  keras.layers.Dense(1),
])
optimizer = keras.optimizers.Nadam(lr=5e-5)

model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Set early stopping to prevent over fitting
early_stopping = keras.callbacks.EarlyStopping(patience=50)

# save best model to load later
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)

# compile model
model.fit(train_set, epochs=500,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint])
```

    Epoch 1/500
         41/Unknown - 1s 16ms/step - loss: 0.0055 - mae: 0.0783WARNING:tensorflow:From C:\Users\blake\Anaconda3\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From C:\Users\blake\Anaconda3\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 3s 60ms/step - loss: 0.0054 - mae: 0.0782 - val_loss: 0.1909 - val_mae: 0.5187
    Epoch 2/500
    40/43 [==========================>...] - ETA: 0s - loss: 0.0013 - mae: 0.0364INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 0.0013 - mae: 0.0361 - val_loss: 0.1030 - val_mae: 0.3616
    Epoch 3/500
    41/43 [===========================>..] - ETA: 0s - loss: 8.1130e-04 - mae: 0.0313INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 8.0252e-04 - mae: 0.0311 - val_loss: 0.0831 - val_mae: 0.3203
    Epoch 4/500
    40/43 [==========================>...] - ETA: 0s - loss: 4.6892e-04 - mae: 0.0224INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 4.7788e-04 - mae: 0.0226 - val_loss: 0.0702 - val_mae: 0.2894
    Epoch 5/500
    42/43 [============================>.] - ETA: 0s - loss: 3.5751e-04 - mae: 0.0195INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 3.7628e-04 - mae: 0.0200 - val_loss: 0.0556 - val_mae: 0.2438
    Epoch 6/500
    41/43 [===========================>..] - ETA: 0s - loss: 5.9585e-04 - mae: 0.0259INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 5.9004e-04 - mae: 0.0258 - val_loss: 0.0537 - val_mae: 0.2443
    Epoch 7/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.6126e-04 - mae: 0.0169INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 2.7250e-04 - mae: 0.0173 - val_loss: 0.0509 - val_mae: 0.2403
    Epoch 8/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.9441e-04 - mae: 0.0181INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 3.0362e-04 - mae: 0.0184 - val_loss: 0.0488 - val_mae: 0.2380
    Epoch 9/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.0457e-04 - mae: 0.0184INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 3.2637e-04 - mae: 0.0191 - val_loss: 0.0417 - val_mae: 0.2096
    Epoch 10/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.3423e-04 - mae: 0.0165INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 2.3843e-04 - mae: 0.0166 - val_loss: 0.0416 - val_mae: 0.2146
    Epoch 11/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.8580e-04 - mae: 0.0218 - val_loss: 0.0424 - val_mae: 0.2231
    Epoch 12/500
    41/43 [===========================>..] - ETA: 0s - loss: 1.8747e-04 - mae: 0.0144INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 1.9867e-04 - mae: 0.0148 - val_loss: 0.0414 - val_mae: 0.2227
    Epoch 13/500
    41/43 [===========================>..] - ETA: 0s - loss: 1.7239e-04 - mae: 0.0139INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 1.7863e-04 - mae: 0.0142 - val_loss: 0.0347 - val_mae: 0.1923
    Epoch 14/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.6806e-04 - mae: 0.0204 - val_loss: 0.0360 - val_mae: 0.2036
    Epoch 15/500
    40/43 [==========================>...] - ETA: 0s - loss: 1.3257e-04 - mae: 0.0120INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 1.3999e-04 - mae: 0.0123 - val_loss: 0.0328 - val_mae: 0.1905
    Epoch 16/500
    41/43 [===========================>..] - ETA: 0s - loss: 1.3353e-04 - mae: 0.0120INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 1.3903e-04 - mae: 0.0123 - val_loss: 0.0285 - val_mae: 0.1702
    Epoch 17/500
    41/43 [===========================>..] - ETA: 0s - loss: 4.1237e-04 - mae: 0.0216INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 4.1094e-04 - mae: 0.0217 - val_loss: 0.0268 - val_mae: 0.1636
    Epoch 18/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.4351e-04 - mae: 0.0125 - val_loss: 0.0289 - val_mae: 0.1789
    Epoch 19/500
    40/43 [==========================>...] - ETA: 0s - loss: 1.1217e-04 - mae: 0.0109INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 1.1299e-04 - mae: 0.0110 - val_loss: 0.0264 - val_mae: 0.1672
    Epoch 20/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.2536e-04 - mae: 0.0116 - val_loss: 0.0272 - val_mae: 0.1749
    Epoch 21/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.3073e-04 - mae: 0.0162INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 2.3286e-04 - mae: 0.0163 - val_loss: 0.0255 - val_mae: 0.1665
    Epoch 22/500
    41/43 [===========================>..] - ETA: 0s - loss: 9.9316e-05 - mae: 0.0103INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 1.0448e-04 - mae: 0.0106 - val_loss: 0.0225 - val_mae: 0.1504
    Epoch 23/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2922e-04 - mae: 0.0165 - val_loss: 0.0253 - val_mae: 0.1706
    Epoch 24/500
    41/43 [===========================>..] - ETA: 0s - loss: 9.4340e-05 - mae: 0.0100INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 9.6946e-05 - mae: 0.0102 - val_loss: 0.0216 - val_mae: 0.1486
    Epoch 25/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.1103e-04 - mae: 0.0110 - val_loss: 0.0220 - val_mae: 0.1543
    Epoch 26/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.5407e-04 - mae: 0.0132 - val_loss: 0.0221 - val_mae: 0.1567
    Epoch 27/500
    41/43 [===========================>..] - ETA: 0s - loss: 9.3717e-05 - mae: 0.0101INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 9.6044e-05 - mae: 0.0102 - val_loss: 0.0204 - val_mae: 0.1467
    Epoch 28/500
    41/43 [===========================>..] - ETA: 0s - loss: 8.9582e-05 - mae: 0.009 - ETA: 0s - loss: 1.2365e-04 - mae: 0.0112INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 52ms/step - loss: 1.5653e-04 - mae: 0.0124 - val_loss: 0.0167 - val_mae: 0.1266
    Epoch 29/500
    43/43 [==============================] - 1s 18ms/step - loss: 1.0441e-04 - mae: 0.0109 - val_loss: 0.0203 - val_mae: 0.1514
    Epoch 30/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.8229e-04 - mae: 0.0175 - val_loss: 0.0201 - val_mae: 0.1496
    Epoch 31/500
    43/43 [==============================] - 1s 18ms/step - loss: 8.0070e-05 - mae: 0.0092 - val_loss: 0.0192 - val_mae: 0.1451
    Epoch 32/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.6521e-05 - mae: 0.0091INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 7.9114e-05 - mae: 0.0092 - val_loss: 0.0164 - val_mae: 0.1270
    Epoch 33/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.5300e-05 - mae: 0.0089INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 7.6168e-05 - mae: 0.0090 - val_loss: 0.0162 - val_mae: 0.1270
    Epoch 34/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.0500e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 7.1373e-05 - mae: 0.0087 - val_loss: 0.0159 - val_mae: 0.1265
    Epoch 35/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7453e-04 - mae: 0.0166 - val_loss: 0.0203 - val_mae: 0.1600
    Epoch 36/500
    42/43 [============================>.] - ETA: 0s - loss: 7.7125e-05 - mae: 0.0092INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 7.8351e-05 - mae: 0.0093 - val_loss: 0.0145 - val_mae: 0.1188
    Epoch 37/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.9617e-04 - mae: 0.0154 - val_loss: 0.0173 - val_mae: 0.1411
    Epoch 38/500
    43/43 [==============================] - ETA: 0s - loss: 7.0999e-05 - mae: 0.0088INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 7.0999e-05 - mae: 0.0088 - val_loss: 0.0139 - val_mae: 0.1165
    Epoch 39/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.6828e-05 - mae: 0.0083INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 55ms/step - loss: 6.8992e-05 - mae: 0.0085 - val_loss: 0.0132 - val_mae: 0.1124
    Epoch 40/500
    43/43 [==============================] - 1s 19ms/step - loss: 7.8104e-05 - mae: 0.0092 - val_loss: 0.0137 - val_mae: 0.1172
    Epoch 41/500
    41/43 [===========================>..] - ETA: 0s - loss: 1.6074e-04 - mae: 0.0137INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 1.6946e-04 - mae: 0.0141 - val_loss: 0.0115 - val_mae: 0.1043
    Epoch 42/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.0613e-04 - mae: 0.0112 - val_loss: 0.0136 - val_mae: 0.1177
    Epoch 43/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.6824e-05 - mae: 0.0084 - val_loss: 0.0128 - val_mae: 0.1119
    Epoch 44/500
    43/43 [==============================] - 1s 18ms/step - loss: 7.6693e-05 - mae: 0.0090 - val_loss: 0.0130 - val_mae: 0.1148
    Epoch 45/500
    40/43 [==========================>...] - ETA: 0s - loss: 1.0978e-04 - mae: 0.0113INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 1.1693e-04 - mae: 0.0117 - val_loss: 0.0110 - val_mae: 0.1015
    Epoch 46/500
    43/43 [==============================] - ETA: 0s - loss: 6.0563e-05 - mae: 0.008 - 1s 19ms/step - loss: 6.5853e-05 - mae: 0.0085 - val_loss: 0.0129 - val_mae: 0.1158
    Epoch 47/500
    43/43 [==============================] - 1s 19ms/step - loss: 7.5388e-05 - mae: 0.0092 - val_loss: 0.0116 - val_mae: 0.1058
    Epoch 48/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.5476e-05 - mae: 0.0077 - val_loss: 0.0117 - val_mae: 0.1068
    Epoch 49/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1323e-04 - mae: 0.0146 - val_loss: 0.0169 - val_mae: 0.1512
    Epoch 50/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.1130e-04 - mae: 0.0113 - val_loss: 0.0111 - val_mae: 0.1043
    Epoch 51/500
    43/43 [==============================] - 1s 19ms/step - loss: 7.6599e-05 - mae: 0.0093 - val_loss: 0.0117 - val_mae: 0.1107
    Epoch 52/500
    42/43 [============================>.] - ETA: 0s - loss: 8.6424e-05 - mae: 0.0097INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 8.7214e-05 - mae: 0.0098 - val_loss: 0.0100 - val_mae: 0.0967
    Epoch 53/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.3385e-05 - mae: 0.0082 - val_loss: 0.0104 - val_mae: 0.1007
    Epoch 54/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.3242e-05 - mae: 0.0075 - val_loss: 0.0100 - val_mae: 0.0979
    Epoch 55/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.6250e-04 - mae: 0.0141 - val_loss: 0.0121 - val_mae: 0.1167
    Epoch 56/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.6163e-05 - mae: 0.0078 - val_loss: 0.0107 - val_mae: 0.1042
    Epoch 57/500
    43/43 [==============================] - 1s 19ms/step - loss: 7.2503e-05 - mae: 0.0088 - val_loss: 0.0107 - val_mae: 0.1057
    Epoch 58/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.2967e-05 - mae: 0.0084INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 6.3143e-05 - mae: 0.0084 - val_loss: 0.0098 - val_mae: 0.0976
    Epoch 59/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.0659e-05 - mae: 0.0081INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 6.1814e-05 - mae: 0.0082 - val_loss: 0.0096 - val_mae: 0.0961
    Epoch 60/500
    41/43 [===========================>..] - ETA: 0s - loss: 6.0025e-05 - mae: 0.0081INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 6.2744e-05 - mae: 0.0084 - val_loss: 0.0087 - val_mae: 0.0897
    Epoch 61/500
    41/43 [===========================>..] - ETA: 0s - loss: 7.2136e-05 - mae: 0.0086INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 8.4165e-05 - mae: 0.0093 - val_loss: 0.0075 - val_mae: 0.0849
    Epoch 62/500
    43/43 [==============================] - 1s 19ms/step - loss: 9.9946e-05 - mae: 0.0109 - val_loss: 0.0108 - val_mae: 0.1110
    Epoch 63/500
    43/43 [==============================] - 1s 19ms/step - loss: 8.9505e-05 - mae: 0.0101 - val_loss: 0.0107 - val_mae: 0.1096
    Epoch 64/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.8270e-05 - mae: 0.0087 - val_loss: 0.0096 - val_mae: 0.0996
    Epoch 65/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.9440e-05 - mae: 0.0073 - val_loss: 0.0091 - val_mae: 0.0961
    Epoch 66/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.2406e-05 - mae: 0.0075 - val_loss: 0.0085 - val_mae: 0.0906
    Epoch 67/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.5968e-04 - mae: 0.0134 - val_loss: 0.0079 - val_mae: 0.0852
    Epoch 68/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.7316e-05 - mae: 0.0072 - val_loss: 0.0080 - val_mae: 0.0863
    Epoch 69/500
    43/43 [==============================] - 1s 19ms/step - loss: 7.9423e-05 - mae: 0.0092 - val_loss: 0.0095 - val_mae: 0.1011
    Epoch 70/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.7020e-05 - mae: 0.0071 - val_loss: 0.0090 - val_mae: 0.0972
    Epoch 71/500
    43/43 [==============================] - 1s 19ms/step - loss: 8.7253e-05 - mae: 0.0101 - val_loss: 0.0077 - val_mae: 0.0842
    Epoch 72/500
    40/43 [==========================>...] - ETA: 0s - loss: 5.3305e-05 - mae: 0.0077INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 6.0296e-05 - mae: 0.0082 - val_loss: 0.0070 - val_mae: 0.0801
    Epoch 73/500
    43/43 [==============================] - 1s 19ms/step - loss: 9.6903e-05 - mae: 0.0105 - val_loss: 0.0094 - val_mae: 0.1019
    Epoch 74/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.1033e-05 - mae: 0.0074 - val_loss: 0.0070 - val_mae: 0.0800
    Epoch 75/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.6357e-05 - mae: 0.0086 - val_loss: 0.0085 - val_mae: 0.0938
    Epoch 76/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.1086e-04 - mae: 0.0114 - val_loss: 0.0090 - val_mae: 0.0990
    Epoch 77/500
    41/43 [===========================>..] - ETA: 0s - loss: 4.4287e-05 - mae: 0.0069INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 4.7366e-05 - mae: 0.0071 - val_loss: 0.0069 - val_mae: 0.0793
    Epoch 78/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.2112e-05 - mae: 0.0067 - val_loss: 0.0077 - val_mae: 0.0865
    Epoch 79/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.3357e-05 - mae: 0.0069 - val_loss: 0.0075 - val_mae: 0.0853
    Epoch 80/500
    41/43 [===========================>..] - ETA: 0s - loss: 5.6561e-05 - mae: 0.0077INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 6.6687e-05 - mae: 0.0083 - val_loss: 0.0060 - val_mae: 0.0755
    Epoch 81/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.0121e-04 - mae: 0.0110 - val_loss: 0.0081 - val_mae: 0.0921
    Epoch 82/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.8091e-05 - mae: 0.0072 - val_loss: 0.0079 - val_mae: 0.0899
    Epoch 83/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.8199e-05 - mae: 0.0064 - val_loss: 0.0067 - val_mae: 0.0782
    Epoch 84/500
    42/43 [============================>.] - ETA: 0s - loss: 8.6396e-05 - mae: 0.0098INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 8.9383e-05 - mae: 0.0099 - val_loss: 0.0060 - val_mae: 0.0740
    Epoch 85/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.4423e-05 - mae: 0.0078 - val_loss: 0.0072 - val_mae: 0.0844
    Epoch 86/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.0182e-05 - mae: 0.0073 - val_loss: 0.0070 - val_mae: 0.0823
    Epoch 87/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.0770e-05 - mae: 0.0066 - val_loss: 0.0071 - val_mae: 0.0828
    Epoch 88/500
    43/43 [==============================] - 1s 18ms/step - loss: 8.8760e-05 - mae: 0.0099 - val_loss: 0.0086 - val_mae: 0.0998
    Epoch 89/500
    43/43 [==============================] - 1s 19ms/step - loss: 9.2456e-05 - mae: 0.0103 - val_loss: 0.0083 - val_mae: 0.0968
    Epoch 90/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.0913e-05 - mae: 0.0067 - val_loss: 0.0064 - val_mae: 0.0773
    Epoch 91/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.3413e-05 - mae: 0.0068 - val_loss: 0.0073 - val_mae: 0.0882
    Epoch 92/500
    40/43 [==========================>...] - ETA: 0s - loss: 5.5075e-05 - mae: 0.0076INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 5.9633e-05 - mae: 0.0080 - val_loss: 0.0056 - val_mae: 0.0712
    Epoch 93/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.5069e-05 - mae: 0.0080 - val_loss: 0.0059 - val_mae: 0.0732
    Epoch 94/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.3722e-05 - mae: 0.0068 - val_loss: 0.0074 - val_mae: 0.0904
    Epoch 95/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.9597e-05 - mae: 0.0074 - val_loss: 0.0073 - val_mae: 0.0897
    Epoch 96/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.3856e-05 - mae: 0.0078 - val_loss: 0.0057 - val_mae: 0.0725
    Epoch 97/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.3601e-05 - mae: 0.0070 - val_loss: 0.0059 - val_mae: 0.0748
    Epoch 98/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.0497e-05 - mae: 0.0066 - val_loss: 0.0070 - val_mae: 0.0876
    Epoch 99/500
    40/43 [==========================>...] - ETA: 0s - loss: 1.2554e-04 - mae: 0.0113INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 1.5047e-04 - mae: 0.0125 - val_loss: 0.0046 - val_mae: 0.0662
    Epoch 100/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.1128e-05 - mae: 0.0084 - val_loss: 0.0055 - val_mae: 0.0724
    Epoch 101/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.1728e-05 - mae: 0.0068 - val_loss: 0.0057 - val_mae: 0.0738
    Epoch 102/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.7438e-05 - mae: 0.0063 - val_loss: 0.0049 - val_mae: 0.0664
    Epoch 103/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.6575e-05 - mae: 0.0063 - val_loss: 0.0055 - val_mae: 0.0716
    Epoch 104/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.2718e-05 - mae: 0.0068 - val_loss: 0.0059 - val_mae: 0.0761
    Epoch 105/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9557e-05 - mae: 0.0065 - val_loss: 0.0063 - val_mae: 0.0810
    Epoch 106/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.2365e-05 - mae: 0.0058 - val_loss: 0.0057 - val_mae: 0.0748
    Epoch 107/500
    43/43 [==============================] - 1s 20ms/step - loss: 4.0070e-05 - mae: 0.0065 - val_loss: 0.0066 - val_mae: 0.0863
    Epoch 108/500
    40/43 [==========================>...] - ETA: 0s - loss: 1.0915e-04 - mae: 0.0116INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 1.2433e-04 - mae: 0.0124 - val_loss: 0.0044 - val_mae: 0.0650
    Epoch 109/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.7175e-05 - mae: 0.0074 - val_loss: 0.0055 - val_mae: 0.0742
    Epoch 110/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.3131e-05 - mae: 0.0059 - val_loss: 0.0055 - val_mae: 0.0739
    Epoch 111/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.8043e-05 - mae: 0.0073 - val_loss: 0.0056 - val_mae: 0.0748
    Epoch 112/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.5067e-05 - mae: 0.0061 - val_loss: 0.0055 - val_mae: 0.0741
    Epoch 113/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9833e-05 - mae: 0.0055 - val_loss: 0.0052 - val_mae: 0.0713
    Epoch 114/500
    43/43 [==============================] - 1s 19ms/step - loss: 1.0657e-04 - mae: 0.0112 - val_loss: 0.0065 - val_mae: 0.0863
    Epoch 115/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.7004e-05 - mae: 0.0063 - val_loss: 0.0050 - val_mae: 0.0688
    Epoch 116/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.4347e-05 - mae: 0.0060 - val_loss: 0.0057 - val_mae: 0.0779
    Epoch 117/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.1342e-05 - mae: 0.0058 - val_loss: 0.0047 - val_mae: 0.0657
    Epoch 118/500
    40/43 [==========================>...] - ETA: 0s - loss: 8.1623e-05 - mae: 0.0097INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 8.9209e-05 - mae: 0.0102 - val_loss: 0.0039 - val_mae: 0.0603
    Epoch 119/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9313e-05 - mae: 0.0065 - val_loss: 0.0049 - val_mae: 0.0702
    Epoch 120/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.0727e-05 - mae: 0.0066 - val_loss: 0.0040 - val_mae: 0.0601
    Epoch 121/500
    43/43 [==============================] - ETA: 0s - loss: 3.7454e-05 - mae: 0.0064INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 58ms/step - loss: 3.7454e-05 - mae: 0.0064 - val_loss: 0.0038 - val_mae: 0.0586
    Epoch 122/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.6077e-05 - mae: 0.0061 - val_loss: 0.0054 - val_mae: 0.0769
    Epoch 123/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.7390e-05 - mae: 0.0074 - val_loss: 0.0052 - val_mae: 0.0740
    Epoch 124/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.8090e-05 - mae: 0.0073 - val_loss: 0.0051 - val_mae: 0.0737
    Epoch 125/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.8961e-05 - mae: 0.0055 - val_loss: 0.0045 - val_mae: 0.0664
    Epoch 126/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.8735e-05 - mae: 0.0063INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 4.8729e-05 - mae: 0.0070 - val_loss: 0.0036 - val_mae: 0.0573
    Epoch 127/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.7915e-05 - mae: 0.0065 - val_loss: 0.0047 - val_mae: 0.0700
    Epoch 128/500
    43/43 [==============================] - 1s 19ms/step - loss: 6.0933e-05 - mae: 0.0083 - val_loss: 0.0036 - val_mae: 0.0573
    Epoch 129/500
    43/43 [==============================] - 1s 20ms/step - loss: 5.5741e-05 - mae: 0.0081 - val_loss: 0.0042 - val_mae: 0.0630
    Epoch 130/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9078e-05 - mae: 0.0055 - val_loss: 0.0044 - val_mae: 0.0653
    Epoch 131/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.8167e-05 - mae: 0.0053 - val_loss: 0.0047 - val_mae: 0.0687
    Epoch 132/500
    43/43 [==============================] - ETA: 0s - loss: 3.7660e-05 - mae: 0.0063INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 3s 58ms/step - loss: 3.7660e-05 - mae: 0.0063 - val_loss: 0.0031 - val_mae: 0.0560
    Epoch 133/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.5303e-05 - mae: 0.0063 - val_loss: 0.0041 - val_mae: 0.0638
    Epoch 134/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.2147e-05 - mae: 0.0086 - val_loss: 0.0051 - val_mae: 0.0749
    Epoch 135/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.4492e-05 - mae: 0.0060INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 4.9435e-05 - mae: 0.0070 - val_loss: 0.0030 - val_mae: 0.0551
    Epoch 136/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.2596e-05 - mae: 0.0070 - val_loss: 0.0030 - val_mae: 0.0519
    Epoch 137/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.4526e-05 - mae: 0.0070 - val_loss: 0.0043 - val_mae: 0.0679
    Epoch 138/500
    43/43 [==============================] - 1s 18ms/step - loss: 6.6383e-05 - mae: 0.0090 - val_loss: 0.0043 - val_mae: 0.0660
    Epoch 139/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.8583e-05 - mae: 0.0054 - val_loss: 0.0043 - val_mae: 0.0661
    Epoch 140/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.8626e-05 - mae: 0.0054 - val_loss: 0.0043 - val_mae: 0.0664
    Epoch 141/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.2389e-05 - mae: 0.0059 - val_loss: 0.0036 - val_mae: 0.0572
    Epoch 142/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.3427e-05 - mae: 0.0074 - val_loss: 0.0050 - val_mae: 0.0766
    Epoch 143/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.2940e-05 - mae: 0.0060 - val_loss: 0.0034 - val_mae: 0.0555
    Epoch 144/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.4137e-05 - mae: 0.0071 - val_loss: 0.0034 - val_mae: 0.0559
    Epoch 145/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.4170e-05 - mae: 0.0071 - val_loss: 0.0030 - val_mae: 0.0525
    Epoch 146/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.8136e-05 - mae: 0.0062 - val_loss: 0.0042 - val_mae: 0.0677
    Epoch 147/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.1146e-05 - mae: 0.0059 - val_loss: 0.0034 - val_mae: 0.0561
    Epoch 148/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.4864e-05 - mae: 0.0060 - val_loss: 0.0043 - val_mae: 0.0690
    Epoch 149/500
    43/43 [==============================] - 1s 20ms/step - loss: 3.2722e-05 - mae: 0.0060 - val_loss: 0.0034 - val_mae: 0.0558
    Epoch 150/500
    41/43 [===========================>..] - ETA: 0s - loss: 4.9706e-05 - mae: 0.0072INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 58ms/step - loss: 5.5843e-05 - mae: 0.0076 - val_loss: 0.0024 - val_mae: 0.0491
    Epoch 151/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.2322e-05 - mae: 0.0059 - val_loss: 0.0032 - val_mae: 0.0547
    Epoch 152/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.2979e-05 - mae: 0.0060 - val_loss: 0.0031 - val_mae: 0.0531
    Epoch 153/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.8355e-05 - mae: 0.0055 - val_loss: 0.0030 - val_mae: 0.0516
    Epoch 154/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6974e-05 - mae: 0.0053 - val_loss: 0.0029 - val_mae: 0.0512
    Epoch 155/500
    43/43 [==============================] - 1s 18ms/step - loss: 7.4013e-05 - mae: 0.0094 - val_loss: 0.0049 - val_mae: 0.0776
    Epoch 156/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.2626e-05 - mae: 0.0060 - val_loss: 0.0037 - val_mae: 0.0611
    Epoch 157/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6731e-05 - mae: 0.0053 - val_loss: 0.0033 - val_mae: 0.0555
    Epoch 158/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.9245e-05 - mae: 0.0056 - val_loss: 0.0032 - val_mae: 0.0547
    Epoch 159/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7570e-05 - mae: 0.0053 - val_loss: 0.0028 - val_mae: 0.0497
    Epoch 160/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.1833e-05 - mae: 0.0068 - val_loss: 0.0036 - val_mae: 0.0611
    Epoch 161/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.0651e-05 - mae: 0.0056 - val_loss: 0.0044 - val_mae: 0.0718
    Epoch 162/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.4582e-05 - mae: 0.0062 - val_loss: 0.0027 - val_mae: 0.0490
    Epoch 163/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6586e-05 - mae: 0.0053 - val_loss: 0.0025 - val_mae: 0.0474
    Epoch 164/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.7617e-05 - mae: 0.0084 - val_loss: 0.0032 - val_mae: 0.0545
    Epoch 165/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.1118e-05 - mae: 0.0058 - val_loss: 0.0035 - val_mae: 0.0588
    Epoch 166/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7367e-05 - mae: 0.0053 - val_loss: 0.0033 - val_mae: 0.0569
    Epoch 167/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.6704e-05 - mae: 0.0061 - val_loss: 0.0048 - val_mae: 0.0783
    Epoch 168/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.7368e-05 - mae: 0.0065 - val_loss: 0.0029 - val_mae: 0.0511
    Epoch 169/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7117e-05 - mae: 0.0053 - val_loss: 0.0035 - val_mae: 0.0613
    Epoch 170/500
    43/43 [==============================] - 1s 18ms/step - loss: 5.2594e-05 - mae: 0.0077 - val_loss: 0.0043 - val_mae: 0.0741
    Epoch 171/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.3739e-05 - mae: 0.0071 - val_loss: 0.0038 - val_mae: 0.0668
    Epoch 172/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.4630e-05 - mae: 0.0060INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 4.0891e-05 - mae: 0.0065 - val_loss: 0.0023 - val_mae: 0.0453
    Epoch 173/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.8550e-05 - mae: 0.0055 - val_loss: 0.0029 - val_mae: 0.0552
    Epoch 174/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.4473e-05 - mae: 0.0062INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 3.7458e-05 - mae: 0.0065 - val_loss: 0.0022 - val_mae: 0.0446
    Epoch 175/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.4428e-05 - mae: 0.0062 - val_loss: 0.0027 - val_mae: 0.0505
    Epoch 176/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.5337e-05 - mae: 0.0061 - val_loss: 0.0031 - val_mae: 0.0564
    Epoch 177/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.1288e-05 - mae: 0.0059 - val_loss: 0.0030 - val_mae: 0.0542
    Epoch 178/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3865e-05 - mae: 0.0049 - val_loss: 0.0027 - val_mae: 0.0502
    Epoch 179/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6314e-05 - mae: 0.0051 - val_loss: 0.0028 - val_mae: 0.0528
    Epoch 180/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.9297e-05 - mae: 0.0067 - val_loss: 0.0036 - val_mae: 0.0637
    Epoch 181/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.8914e-05 - mae: 0.0055 - val_loss: 0.0022 - val_mae: 0.0450
    Epoch 182/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9308e-05 - mae: 0.0065 - val_loss: 0.0037 - val_mae: 0.0682
    Epoch 183/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.9245e-05 - mae: 0.0068 - val_loss: 0.0026 - val_mae: 0.0491
    Epoch 184/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.9415e-05 - mae: 0.0055INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 3.1448e-05 - mae: 0.0057 - val_loss: 0.0019 - val_mae: 0.0423
    Epoch 185/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.8868e-05 - mae: 0.0064 - val_loss: 0.0031 - val_mae: 0.0599
    Epoch 186/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9413e-05 - mae: 0.0056 - val_loss: 0.0027 - val_mae: 0.0520
    Epoch 187/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.1016e-05 - mae: 0.0057 - val_loss: 0.0027 - val_mae: 0.0530
    Epoch 188/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5679e-05 - mae: 0.0052 - val_loss: 0.0020 - val_mae: 0.0425
    Epoch 189/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5777e-05 - mae: 0.0051 - val_loss: 0.0020 - val_mae: 0.0428
    Epoch 190/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5857e-05 - mae: 0.0052 - val_loss: 0.0021 - val_mae: 0.0431
    Epoch 191/500
    42/43 [============================>.] - ETA: 0s - loss: 7.4851e-05 - mae: 0.0088INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 7.6234e-05 - mae: 0.0090 - val_loss: 0.0018 - val_mae: 0.0409
    Epoch 192/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5449e-05 - mae: 0.0052 - val_loss: 0.0019 - val_mae: 0.0411
    Epoch 193/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7245e-05 - mae: 0.0053 - val_loss: 0.0023 - val_mae: 0.0466
    Epoch 194/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.2364e-05 - mae: 0.0058 - val_loss: 0.0029 - val_mae: 0.0569
    Epoch 195/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5461e-05 - mae: 0.0051 - val_loss: 0.0021 - val_mae: 0.0433
    Epoch 196/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.4822e-05 - mae: 0.0050 - val_loss: 0.0021 - val_mae: 0.0429
    Epoch 197/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6749e-05 - mae: 0.0053 - val_loss: 0.0025 - val_mae: 0.0489
    Epoch 198/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9015e-05 - mae: 0.0056 - val_loss: 0.0026 - val_mae: 0.0517
    Epoch 199/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.3340e-05 - mae: 0.0069 - val_loss: 0.0019 - val_mae: 0.0417
    Epoch 200/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9239e-05 - mae: 0.0055 - val_loss: 0.0030 - val_mae: 0.0597
    Epoch 201/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7307e-05 - mae: 0.0053 - val_loss: 0.0026 - val_mae: 0.0518
    Epoch 202/500
    43/43 [==============================] - ETA: 0s - loss: 5.2539e-05 - mae: 0.0076INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 5.2539e-05 - mae: 0.0076 - val_loss: 0.0018 - val_mae: 0.0407
    Epoch 203/500
    43/43 [==============================] - 1s 20ms/step - loss: 3.0762e-05 - mae: 0.0057 - val_loss: 0.0024 - val_mae: 0.0497
    Epoch 204/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.6633e-05 - mae: 0.0052 - val_loss: 0.0024 - val_mae: 0.0496
    Epoch 205/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.6053e-05 - mae: 0.0051 - val_loss: 0.0021 - val_mae: 0.0450
    Epoch 206/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.5428e-05 - mae: 0.0051 - val_loss: 0.0019 - val_mae: 0.0415
    Epoch 207/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5405e-05 - mae: 0.0050 - val_loss: 0.0025 - val_mae: 0.0508
    Epoch 208/500
    43/43 [==============================] - ETA: 0s - loss: 4.0634e-05 - mae: 0.0067INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 4.0634e-05 - mae: 0.0067 - val_loss: 0.0018 - val_mae: 0.0407
    Epoch 209/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.5317e-05 - mae: 0.0052 - val_loss: 0.0020 - val_mae: 0.0435
    Epoch 210/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.8747e-05 - mae: 0.0055 - val_loss: 0.0024 - val_mae: 0.0481
    Epoch 211/500
    40/43 [==========================>...] - ETA: 0s - loss: 3.2340e-05 - mae: 0.0058INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 58ms/step - loss: 3.3822e-05 - mae: 0.0060 - val_loss: 0.0017 - val_mae: 0.0399
    Epoch 212/500
    40/43 [==========================>...] - ETA: 0s - loss: 2.5942e-05 - mae: 0.0052INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 3.0419e-05 - mae: 0.0056 - val_loss: 0.0015 - val_mae: 0.0382
    Epoch 213/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5004e-05 - mae: 0.0051 - val_loss: 0.0018 - val_mae: 0.0395
    Epoch 214/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.8520e-05 - mae: 0.0055 - val_loss: 0.0018 - val_mae: 0.0405
    Epoch 215/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5289e-05 - mae: 0.0051 - val_loss: 0.0026 - val_mae: 0.0532
    Epoch 216/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.5195e-05 - mae: 0.0060 - val_loss: 0.0016 - val_mae: 0.0397
    Epoch 217/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9805e-05 - mae: 0.0058 - val_loss: 0.0020 - val_mae: 0.0453
    Epoch 218/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.4743e-05 - mae: 0.0063 - val_loss: 0.0021 - val_mae: 0.0455
    Epoch 219/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7327e-05 - mae: 0.0052 - val_loss: 0.0024 - val_mae: 0.0501
    Epoch 220/500
    43/43 [==============================] - ETA: 0s - loss: 3.6649e-05 - mae: 0.0060INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 55ms/step - loss: 3.6649e-05 - mae: 0.0060 - val_loss: 0.0014 - val_mae: 0.0373
    Epoch 221/500
    43/43 [==============================] - 1s 21ms/step - loss: 2.7475e-05 - mae: 0.0054 - val_loss: 0.0018 - val_mae: 0.0421
    Epoch 222/500
    43/43 [==============================] - 1s 20ms/step - loss: 4.4651e-05 - mae: 0.0070 - val_loss: 0.0022 - val_mae: 0.0466
    Epoch 223/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.6085e-05 - mae: 0.0051 - val_loss: 0.0019 - val_mae: 0.0418
    Epoch 224/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.5407e-05 - mae: 0.0050 - val_loss: 0.0019 - val_mae: 0.0420
    Epoch 225/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3945e-05 - mae: 0.0049 - val_loss: 0.0017 - val_mae: 0.0397
    Epoch 226/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2177e-05 - mae: 0.0046 - val_loss: 0.0017 - val_mae: 0.0397
    Epoch 227/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1687e-05 - mae: 0.0046 - val_loss: 0.0020 - val_mae: 0.0441
    Epoch 228/500
    43/43 [==============================] - 1s 19ms/step - loss: 4.3430e-05 - mae: 0.0066 - val_loss: 0.0035 - val_mae: 0.0685
    Epoch 229/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.5103e-05 - mae: 0.0064 - val_loss: 0.0023 - val_mae: 0.0510
    Epoch 230/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2721e-05 - mae: 0.0048 - val_loss: 0.0017 - val_mae: 0.0416
    Epoch 231/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.2880e-05 - mae: 0.0048 - val_loss: 0.0016 - val_mae: 0.0378
    Epoch 232/500
    43/43 [==============================] - ETA: 0s - loss: 2.5408e-05 - mae: 0.0051INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 2.5408e-05 - mae: 0.0051 - val_loss: 0.0013 - val_mae: 0.0344
    Epoch 233/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9068e-05 - mae: 0.0055 - val_loss: 0.0019 - val_mae: 0.0449
    Epoch 234/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3927e-05 - mae: 0.0050 - val_loss: 0.0015 - val_mae: 0.0362
    Epoch 235/500
    43/43 [==============================] - 1s 19ms/step - loss: 5.7891e-05 - mae: 0.0079 - val_loss: 0.0031 - val_mae: 0.0656
    Epoch 236/500
    43/43 [==============================] - 1s 20ms/step - loss: 4.4709e-05 - mae: 0.0073 - val_loss: 0.0016 - val_mae: 0.0392
    Epoch 237/500
    43/43 [==============================] - 1s 21ms/step - loss: 2.2467e-05 - mae: 0.0048 - val_loss: 0.0015 - val_mae: 0.0375
    Epoch 238/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.1319e-05 - mae: 0.0045 - val_loss: 0.0015 - val_mae: 0.0384
    Epoch 239/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.3120e-05 - mae: 0.0048 - val_loss: 0.0015 - val_mae: 0.0376
    Epoch 240/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.1969e-05 - mae: 0.0046 - val_loss: 0.0014 - val_mae: 0.0363
    Epoch 241/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.4297e-05 - mae: 0.0049 - val_loss: 0.0016 - val_mae: 0.0388
    Epoch 242/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.7740e-05 - mae: 0.0052 - val_loss: 0.0017 - val_mae: 0.0398
    Epoch 243/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.6351e-05 - mae: 0.0052 - val_loss: 0.0021 - val_mae: 0.0484
    Epoch 244/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.6713e-05 - mae: 0.0052 - val_loss: 0.0020 - val_mae: 0.0455
    Epoch 245/500
    40/43 [==========================>...] - ETA: 0s - loss: 2.3011e-05 - mae: 0.0047INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 2.9583e-05 - mae: 0.0053 - val_loss: 0.0011 - val_mae: 0.0338
    Epoch 246/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.7329e-05 - mae: 0.0054INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 58ms/step - loss: 2.8082e-05 - mae: 0.0055 - val_loss: 0.0011 - val_mae: 0.0312
    Epoch 247/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9221e-05 - mae: 0.0055 - val_loss: 0.0012 - val_mae: 0.0329
    Epoch 248/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.6005e-05 - mae: 0.0052 - val_loss: 0.0016 - val_mae: 0.0388
    Epoch 249/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.1981e-05 - mae: 0.0057 - val_loss: 0.0023 - val_mae: 0.0525
    Epoch 250/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5437e-05 - mae: 0.0051 - val_loss: 0.0019 - val_mae: 0.0444
    Epoch 251/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3003e-05 - mae: 0.0048 - val_loss: 0.0014 - val_mae: 0.0356
    Epoch 252/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.4438e-05 - mae: 0.0050 - val_loss: 0.0012 - val_mae: 0.0334
    Epoch 253/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.0598e-05 - mae: 0.0058 - val_loss: 0.0019 - val_mae: 0.0448
    Epoch 254/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.6391e-05 - mae: 0.0052 - val_loss: 0.0019 - val_mae: 0.0445
    Epoch 255/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.8940e-05 - mae: 0.0054 - val_loss: 0.0013 - val_mae: 0.0343
    Epoch 256/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9538e-05 - mae: 0.0058 - val_loss: 0.0016 - val_mae: 0.0388
    Epoch 257/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3903e-05 - mae: 0.0048 - val_loss: 0.0012 - val_mae: 0.0335
    Epoch 258/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.3862e-05 - mae: 0.0049 - val_loss: 0.0015 - val_mae: 0.0382
    Epoch 259/500
    41/43 [===========================>..] - ETA: 0s - loss: 3.0268e-05 - mae: 0.0053INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 3.4568e-05 - mae: 0.0057 - val_loss: 0.0011 - val_mae: 0.0329
    Epoch 260/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.4837e-05 - mae: 0.0063 - val_loss: 0.0013 - val_mae: 0.0341
    Epoch 261/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3209e-05 - mae: 0.0048 - val_loss: 0.0017 - val_mae: 0.0409
    Epoch 262/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1776e-05 - mae: 0.0046 - val_loss: 0.0016 - val_mae: 0.0394
    Epoch 263/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0582e-05 - mae: 0.0044 - val_loss: 0.0015 - val_mae: 0.0371
    Epoch 264/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.3276e-05 - mae: 0.0058 - val_loss: 0.0024 - val_mae: 0.0548
    Epoch 265/500
    43/43 [==============================] - 1s 20ms/step - loss: 3.8694e-05 - mae: 0.0064 - val_loss: 0.0014 - val_mae: 0.0357
    Epoch 266/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3689e-05 - mae: 0.0049 - val_loss: 0.0013 - val_mae: 0.0352
    Epoch 267/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2199e-05 - mae: 0.0046 - val_loss: 0.0015 - val_mae: 0.0379
    Epoch 268/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.6647e-05 - mae: 0.0052 - val_loss: 0.0016 - val_mae: 0.0415
    Epoch 269/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.4093e-05 - mae: 0.0050 - val_loss: 0.0016 - val_mae: 0.0398
    Epoch 270/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7333e-05 - mae: 0.0052 - val_loss: 0.0018 - val_mae: 0.0449
    Epoch 271/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.5603e-05 - mae: 0.0052 - val_loss: 0.0014 - val_mae: 0.0358
    Epoch 272/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.2421e-05 - mae: 0.0047 - val_loss: 0.0015 - val_mae: 0.0379
    Epoch 273/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.1416e-05 - mae: 0.0045 - val_loss: 0.0013 - val_mae: 0.0349
    Epoch 274/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.1617e-05 - mae: 0.0056 - val_loss: 0.0018 - val_mae: 0.0447
    Epoch 275/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7214e-05 - mae: 0.0053 - val_loss: 0.0011 - val_mae: 0.0318
    Epoch 276/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7809e-05 - mae: 0.0054 - val_loss: 0.0016 - val_mae: 0.0419
    Epoch 277/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6672e-05 - mae: 0.0052 - val_loss: 0.0016 - val_mae: 0.0415
    Epoch 278/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.0964e-05 - mae: 0.0057 - val_loss: 0.0011 - val_mae: 0.0314
    Epoch 279/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6208e-05 - mae: 0.0051 - val_loss: 0.0012 - val_mae: 0.0325
    Epoch 280/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3725e-05 - mae: 0.0049 - val_loss: 0.0014 - val_mae: 0.0361
    Epoch 281/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5316e-05 - mae: 0.0050 - val_loss: 0.0018 - val_mae: 0.0444
    Epoch 282/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4450e-05 - mae: 0.0050 - val_loss: 0.0015 - val_mae: 0.0381
    Epoch 283/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4741e-05 - mae: 0.0050 - val_loss: 0.0018 - val_mae: 0.0453
    Epoch 284/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5659e-05 - mae: 0.0051 - val_loss: 0.0015 - val_mae: 0.0384
    Epoch 285/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0573e-05 - mae: 0.0044 - val_loss: 0.0014 - val_mae: 0.0360
    Epoch 286/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6638e-05 - mae: 0.0051 - val_loss: 0.0011 - val_mae: 0.0323
    Epoch 287/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.6627e-05 - mae: 0.0061 - val_loss: 0.0020 - val_mae: 0.0490
    Epoch 288/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3791e-05 - mae: 0.0049 - val_loss: 0.0015 - val_mae: 0.0380
    Epoch 289/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3415e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0314
    Epoch 290/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4510e-05 - mae: 0.0050 - val_loss: 0.0014 - val_mae: 0.0364
    Epoch 291/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2545e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0351
    Epoch 292/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1804e-05 - mae: 0.0046 - val_loss: 0.0014 - val_mae: 0.0362
    Epoch 293/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1421e-05 - mae: 0.0045 - val_loss: 0.0015 - val_mae: 0.0392
    Epoch 294/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3999e-05 - mae: 0.0049 - val_loss: 0.0012 - val_mae: 0.0330
    Epoch 295/500
    41/43 [===========================>..] - ETA: 0s - loss: 2.7320e-05 - mae: 0.0052INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 3.2057e-05 - mae: 0.0057 - val_loss: 9.1969e-04 - val_mae: 0.0304
    Epoch 296/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4744e-05 - mae: 0.0051 - val_loss: 0.0011 - val_mae: 0.0306
    Epoch 297/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4062e-05 - mae: 0.0049 - val_loss: 0.0015 - val_mae: 0.0393
    Epoch 298/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7747e-05 - mae: 0.0052 - val_loss: 0.0010 - val_mae: 0.0311
    Epoch 299/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4098e-05 - mae: 0.0050 - val_loss: 0.0011 - val_mae: 0.0320
    Epoch 300/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0619e-05 - mae: 0.0044 - val_loss: 0.0014 - val_mae: 0.0367
    Epoch 301/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2838e-05 - mae: 0.0048 - val_loss: 0.0018 - val_mae: 0.0455
    Epoch 302/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2300e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0351
    Epoch 303/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9851e-05 - mae: 0.0054 - val_loss: 0.0023 - val_mae: 0.0560
    Epoch 304/500
    43/43 [==============================] - 1s 18ms/step - loss: 4.4147e-05 - mae: 0.0072 - val_loss: 0.0017 - val_mae: 0.0445
    Epoch 305/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2755e-05 - mae: 0.0048 - val_loss: 0.0013 - val_mae: 0.0368
    Epoch 306/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2457e-05 - mae: 0.0047 - val_loss: 0.0012 - val_mae: 0.0349
    Epoch 307/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1302e-05 - mae: 0.0045 - val_loss: 0.0011 - val_mae: 0.0332
    Epoch 308/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1915e-05 - mae: 0.0046 - val_loss: 0.0014 - val_mae: 0.0380
    Epoch 309/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.0644e-05 - mae: 0.0057 - val_loss: 0.0010 - val_mae: 0.0302
    Epoch 310/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3561e-05 - mae: 0.0048 - val_loss: 0.0015 - val_mae: 0.0413
    Epoch 311/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4160e-05 - mae: 0.0049 - val_loss: 9.6526e-04 - val_mae: 0.0299
    Epoch 312/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2843e-05 - mae: 0.0049 - val_loss: 0.0011 - val_mae: 0.0329
    Epoch 313/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2853e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0368
    Epoch 314/500
    41/43 [===========================>..] - ETA: 0s - loss: 3.3374e-05 - mae: 0.0057INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 3.6320e-05 - mae: 0.0060 - val_loss: 8.3028e-04 - val_mae: 0.0280
    Epoch 315/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.9503e-05 - mae: 0.0056 - val_loss: 0.0010 - val_mae: 0.0302
    Epoch 316/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1988e-05 - mae: 0.0046 - val_loss: 0.0013 - val_mae: 0.0356
    Epoch 317/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2427e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0348
    Epoch 318/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1709e-05 - mae: 0.0046 - val_loss: 0.0011 - val_mae: 0.0318
    Epoch 319/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4514e-05 - mae: 0.0049 - val_loss: 0.0011 - val_mae: 0.0309
    Epoch 320/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2955e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0316
    Epoch 321/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5333e-05 - mae: 0.0050 - val_loss: 8.9157e-04 - val_mae: 0.0289
    Epoch 322/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4352e-05 - mae: 0.0050 - val_loss: 0.0014 - val_mae: 0.0395
    Epoch 323/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3296e-05 - mae: 0.0049 - val_loss: 0.0013 - val_mae: 0.0359
    Epoch 324/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0788e-05 - mae: 0.0044 - val_loss: 0.0011 - val_mae: 0.0313
    Epoch 325/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0811e-05 - mae: 0.0045 - val_loss: 0.0013 - val_mae: 0.0353
    Epoch 326/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5819e-05 - mae: 0.0051 - val_loss: 0.0015 - val_mae: 0.0410
    Epoch 327/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5139e-05 - mae: 0.0051 - val_loss: 9.4334e-04 - val_mae: 0.0294
    Epoch 328/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3166e-05 - mae: 0.0049 - val_loss: 9.7251e-04 - val_mae: 0.0295
    Epoch 329/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.3495e-05 - mae: 0.0058 - val_loss: 8.4405e-04 - val_mae: 0.0288
    Epoch 330/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4147e-05 - mae: 0.0051 - val_loss: 0.0011 - val_mae: 0.0328
    Epoch 331/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1282e-05 - mae: 0.0046 - val_loss: 0.0014 - val_mae: 0.0374
    Epoch 332/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1235e-05 - mae: 0.0045 - val_loss: 0.0011 - val_mae: 0.0325
    Epoch 333/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7070e-05 - mae: 0.0052 - val_loss: 9.9555e-04 - val_mae: 0.0301
    Epoch 334/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7956e-05 - mae: 0.0054 - val_loss: 9.8968e-04 - val_mae: 0.0299
    Epoch 335/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2291e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0371
    Epoch 336/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2722e-05 - mae: 0.0047 - val_loss: 0.0011 - val_mae: 0.0317
    Epoch 337/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1447e-05 - mae: 0.0046 - val_loss: 0.0015 - val_mae: 0.0417
    Epoch 338/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6936e-05 - mae: 0.0052 - val_loss: 0.0017 - val_mae: 0.0459
    Epoch 339/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7704e-05 - mae: 0.0054 - val_loss: 8.7777e-04 - val_mae: 0.0285
    Epoch 340/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1944e-05 - mae: 0.0046 - val_loss: 9.8426e-04 - val_mae: 0.0301
    Epoch 341/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0792e-05 - mae: 0.0045 - val_loss: 0.0012 - val_mae: 0.0352
    Epoch 342/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2400e-05 - mae: 0.0047 - val_loss: 8.3693e-04 - val_mae: 0.0281
    Epoch 343/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7327e-05 - mae: 0.0054 - val_loss: 0.0013 - val_mae: 0.0373
    Epoch 344/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4218e-05 - mae: 0.0049 - val_loss: 0.0014 - val_mae: 0.0381
    Epoch 345/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3154e-05 - mae: 0.0048 - val_loss: 0.0013 - val_mae: 0.0361
    Epoch 346/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2552e-05 - mae: 0.0047 - val_loss: 0.0011 - val_mae: 0.0320
    Epoch 347/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4000e-05 - mae: 0.0049 - val_loss: 0.0012 - val_mae: 0.0337
    Epoch 348/500
    40/43 [==========================>...] - ETA: 0s - loss: 2.0311e-05 - mae: 0.0044INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 2.3166e-05 - mae: 0.0047 - val_loss: 8.2460e-04 - val_mae: 0.0279
    Epoch 349/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3450e-05 - mae: 0.0050 - val_loss: 9.6434e-04 - val_mae: 0.0296
    Epoch 350/500
    43/43 [==============================] - ETA: 0s - loss: 2.3707e-05 - mae: 0.0048INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 57ms/step - loss: 2.3707e-05 - mae: 0.0048 - val_loss: 8.0504e-04 - val_mae: 0.0278
    Epoch 351/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6366e-05 - mae: 0.0053 - val_loss: 0.0012 - val_mae: 0.0352
    Epoch 352/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.7277e-05 - mae: 0.0054 - val_loss: 0.0015 - val_mae: 0.0406
    Epoch 353/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1578e-05 - mae: 0.0046 - val_loss: 0.0010 - val_mae: 0.0310
    Epoch 354/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2899e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0324
    Epoch 355/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0516e-05 - mae: 0.0044 - val_loss: 0.0012 - val_mae: 0.0335
    Epoch 356/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.9928e-05 - mae: 0.0054 - val_loss: 0.0016 - val_mae: 0.0441
    Epoch 357/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4952e-05 - mae: 0.0052 - val_loss: 9.7675e-04 - val_mae: 0.0302
    Epoch 358/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1580e-05 - mae: 0.0046 - val_loss: 8.9492e-04 - val_mae: 0.0285
    Epoch 359/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4318e-05 - mae: 0.0050 - val_loss: 0.0011 - val_mae: 0.0330
    Epoch 360/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2860e-05 - mae: 0.0047 - val_loss: 0.0012 - val_mae: 0.0344
    Epoch 361/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.6666e-05 - mae: 0.0051 - val_loss: 0.0014 - val_mae: 0.0393
    Epoch 362/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4210e-05 - mae: 0.0050 - val_loss: 8.5249e-04 - val_mae: 0.0279
    Epoch 363/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2643e-05 - mae: 0.0047 - val_loss: 9.0418e-04 - val_mae: 0.0287
    Epoch 364/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3048e-05 - mae: 0.0047 - val_loss: 0.0011 - val_mae: 0.0319
    Epoch 365/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2592e-05 - mae: 0.0047 - val_loss: 0.0012 - val_mae: 0.0356
    Epoch 366/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1791e-05 - mae: 0.0046 - val_loss: 0.0013 - val_mae: 0.0375
    Epoch 367/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2399e-05 - mae: 0.0048 - val_loss: 0.0013 - val_mae: 0.0367
    Epoch 368/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0594e-05 - mae: 0.0044 - val_loss: 0.0010 - val_mae: 0.0317
    Epoch 369/500
    41/43 [===========================>..] - ETA: 0s - loss: 3.1947e-05 - mae: 0.0055INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 3.6378e-05 - mae: 0.0059 - val_loss: 6.5580e-04 - val_mae: 0.0262
    Epoch 370/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5152e-05 - mae: 0.0051 - val_loss: 0.0010 - val_mae: 0.0319
    Epoch 371/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2157e-05 - mae: 0.0047 - val_loss: 0.0012 - val_mae: 0.0347
    Epoch 372/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1359e-05 - mae: 0.0045 - val_loss: 0.0011 - val_mae: 0.0335
    Epoch 373/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3106e-05 - mae: 0.0047 - val_loss: 0.0011 - val_mae: 0.0333
    Epoch 374/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0372e-05 - mae: 0.0044 - val_loss: 9.3044e-04 - val_mae: 0.0293
    Epoch 375/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1350e-05 - mae: 0.0046 - val_loss: 0.0010 - val_mae: 0.0311
    Epoch 376/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0707e-05 - mae: 0.0044 - val_loss: 0.0010 - val_mae: 0.0316
    Epoch 377/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3425e-05 - mae: 0.0048 - val_loss: 7.6395e-04 - val_mae: 0.0271
    Epoch 378/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3085e-05 - mae: 0.0049 - val_loss: 7.5990e-04 - val_mae: 0.0264
    Epoch 379/500
    43/43 [==============================] - 1s 18ms/step - loss: 3.7871e-05 - mae: 0.0062 - val_loss: 0.0015 - val_mae: 0.0416
    Epoch 380/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4869e-05 - mae: 0.0051 - val_loss: 9.2698e-04 - val_mae: 0.0295
    Epoch 381/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2519e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0387
    Epoch 382/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2389e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0337
    Epoch 383/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1527e-05 - mae: 0.0046 - val_loss: 0.0012 - val_mae: 0.0359
    Epoch 384/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4842e-05 - mae: 0.0049 - val_loss: 0.0011 - val_mae: 0.0357
    Epoch 385/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2158e-05 - mae: 0.0047 - val_loss: 9.9193e-04 - val_mae: 0.0320
    Epoch 386/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3240e-05 - mae: 0.0048 - val_loss: 9.8955e-04 - val_mae: 0.0320
    Epoch 387/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4613e-05 - mae: 0.0049 - val_loss: 0.0011 - val_mae: 0.0356
    Epoch 388/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1799e-05 - mae: 0.0047 - val_loss: 0.0011 - val_mae: 0.0346
    Epoch 389/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6232e-05 - mae: 0.0051 - val_loss: 0.0011 - val_mae: 0.0351
    Epoch 390/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1096e-05 - mae: 0.0045 - val_loss: 9.9794e-04 - val_mae: 0.0322
    Epoch 391/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3896e-05 - mae: 0.0048 - val_loss: 7.5104e-04 - val_mae: 0.0263
    Epoch 392/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1550e-05 - mae: 0.0046 - val_loss: 7.3705e-04 - val_mae: 0.0259
    Epoch 393/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1337e-05 - mae: 0.0046 - val_loss: 7.5012e-04 - val_mae: 0.0260
    Epoch 394/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4412e-05 - mae: 0.0049 - val_loss: 7.6128e-04 - val_mae: 0.0263
    Epoch 395/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1793e-05 - mae: 0.0047 - val_loss: 0.0012 - val_mae: 0.0366
    Epoch 396/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.2067e-05 - mae: 0.0056 - val_loss: 0.0014 - val_mae: 0.0407
    Epoch 397/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3120e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0354
    Epoch 398/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1377e-05 - mae: 0.0045 - val_loss: 9.9186e-04 - val_mae: 0.0324
    Epoch 399/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0809e-05 - mae: 0.0044 - val_loss: 9.4828e-04 - val_mae: 0.0312
    Epoch 400/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2733e-05 - mae: 0.0047 - val_loss: 8.7290e-04 - val_mae: 0.0289
    Epoch 401/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1083e-05 - mae: 0.0045 - val_loss: 0.0010 - val_mae: 0.0331
    Epoch 402/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2985e-05 - mae: 0.0047 - val_loss: 8.0742e-04 - val_mae: 0.0274
    Epoch 403/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3109e-05 - mae: 0.0049 - val_loss: 0.0011 - val_mae: 0.0345
    Epoch 404/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1268e-05 - mae: 0.0046 - val_loss: 9.7161e-04 - val_mae: 0.0311
    Epoch 405/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2480e-05 - mae: 0.0047 - val_loss: 7.6943e-04 - val_mae: 0.0266
    Epoch 406/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4329e-05 - mae: 0.0049 - val_loss: 0.0012 - val_mae: 0.0355
    Epoch 407/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1367e-05 - mae: 0.0046 - val_loss: 0.0011 - val_mae: 0.0342
    Epoch 408/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3848e-05 - mae: 0.0049 - val_loss: 0.0011 - val_mae: 0.0335
    Epoch 409/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3626e-05 - mae: 0.0049 - val_loss: 8.1432e-04 - val_mae: 0.0275
    Epoch 410/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1082e-05 - mae: 0.0046 - val_loss: 9.3605e-04 - val_mae: 0.0300
    Epoch 411/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5050e-05 - mae: 0.0050 - val_loss: 8.2845e-04 - val_mae: 0.0276
    Epoch 412/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3847e-05 - mae: 0.0048 - val_loss: 9.1321e-04 - val_mae: 0.0292
    Epoch 413/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0950e-05 - mae: 0.0045 - val_loss: 9.6360e-04 - val_mae: 0.0305
    Epoch 414/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0227e-05 - mae: 0.0044 - val_loss: 9.7262e-04 - val_mae: 0.0307
    Epoch 415/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0455e-05 - mae: 0.0044 - val_loss: 0.0012 - val_mae: 0.0355
    Epoch 416/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3646e-05 - mae: 0.0048 - val_loss: 0.0015 - val_mae: 0.0429
    Epoch 417/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.2445e-05 - mae: 0.0048 - val_loss: 0.0010 - val_mae: 0.0320
    Epoch 418/500
    40/43 [==========================>...] - ETA: 0s - loss: 2.2904e-05 - mae: 0.0047INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 56ms/step - loss: 2.5510e-05 - mae: 0.0050 - val_loss: 5.9874e-04 - val_mae: 0.0243
    Epoch 419/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.6752e-05 - mae: 0.0053 - val_loss: 0.0011 - val_mae: 0.0349
    Epoch 420/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4532e-05 - mae: 0.0050 - val_loss: 9.3836e-04 - val_mae: 0.0306
    Epoch 421/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1558e-05 - mae: 0.0045 - val_loss: 8.7726e-04 - val_mae: 0.0291
    Epoch 422/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.5801e-05 - mae: 0.0050 - val_loss: 0.0013 - val_mae: 0.0408
    Epoch 423/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.4982e-05 - mae: 0.0051 - val_loss: 0.0010 - val_mae: 0.0339
    Epoch 424/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2633e-05 - mae: 0.0047 - val_loss: 7.1031e-04 - val_mae: 0.0257
    Epoch 425/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0667e-05 - mae: 0.0045 - val_loss: 9.5963e-04 - val_mae: 0.0318
    Epoch 426/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1446e-05 - mae: 0.0045 - val_loss: 7.7198e-04 - val_mae: 0.0268
    Epoch 427/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0655e-05 - mae: 0.0044 - val_loss: 6.9468e-04 - val_mae: 0.0251
    Epoch 428/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1837e-05 - mae: 0.0045 - val_loss: 0.0010 - val_mae: 0.0330
    Epoch 429/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.4138e-05 - mae: 0.0049 - val_loss: 0.0013 - val_mae: 0.0389
    Epoch 430/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1643e-05 - mae: 0.0046 - val_loss: 8.9672e-04 - val_mae: 0.0298
    Epoch 431/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.4309e-05 - mae: 0.0049 - val_loss: 8.5656e-04 - val_mae: 0.0289
    Epoch 432/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1143e-05 - mae: 0.0046 - val_loss: 0.0011 - val_mae: 0.0342
    Epoch 433/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1524e-05 - mae: 0.0046 - val_loss: 8.4743e-04 - val_mae: 0.0285
    Epoch 434/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.3724e-05 - mae: 0.0048 - val_loss: 0.0012 - val_mae: 0.0373
    Epoch 435/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3179e-05 - mae: 0.0049 - val_loss: 7.2825e-04 - val_mae: 0.0261
    Epoch 436/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2971e-05 - mae: 0.0047 - val_loss: 8.3985e-04 - val_mae: 0.0285
    Epoch 437/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0747e-05 - mae: 0.0044 - val_loss: 8.5603e-04 - val_mae: 0.0286
    Epoch 438/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2337e-05 - mae: 0.0046 - val_loss: 8.7198e-04 - val_mae: 0.0290
    Epoch 439/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0586e-05 - mae: 0.0044 - val_loss: 7.7045e-04 - val_mae: 0.0267
    Epoch 440/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0570e-05 - mae: 0.0044 - val_loss: 8.4017e-04 - val_mae: 0.0280
    Epoch 441/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2814e-05 - mae: 0.0047 - val_loss: 0.0013 - val_mae: 0.0385
    Epoch 442/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3015e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0354
    Epoch 443/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.1344e-05 - mae: 0.0045 - val_loss: 9.8175e-04 - val_mae: 0.0324
    Epoch 444/500
    43/43 [==============================] - 1s 19ms/step - loss: 3.0653e-05 - mae: 0.0055 - val_loss: 0.0012 - val_mae: 0.0380
    Epoch 445/500
    43/43 [==============================] - 1s 20ms/step - loss: 2.4978e-05 - mae: 0.0052 - val_loss: 6.8596e-04 - val_mae: 0.0255
    Epoch 446/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1330e-05 - mae: 0.0046 - val_loss: 9.8653e-04 - val_mae: 0.0330
    Epoch 447/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.7035e-05 - mae: 0.0053 - val_loss: 6.6067e-04 - val_mae: 0.0248
    Epoch 448/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3574e-05 - mae: 0.0048 - val_loss: 8.6745e-04 - val_mae: 0.0299
    Epoch 449/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0421e-05 - mae: 0.0044 - val_loss: 8.0131e-04 - val_mae: 0.0278
    Epoch 450/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2984e-05 - mae: 0.0047 - val_loss: 7.5618e-04 - val_mae: 0.0269
    Epoch 451/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0558e-05 - mae: 0.0044 - val_loss: 8.1714e-04 - val_mae: 0.0281
    Epoch 452/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2676e-05 - mae: 0.0047 - val_loss: 0.0012 - val_mae: 0.0382
    Epoch 453/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2139e-05 - mae: 0.0047 - val_loss: 8.6114e-04 - val_mae: 0.0298
    Epoch 454/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3617e-05 - mae: 0.0048 - val_loss: 6.7958e-04 - val_mae: 0.0252
    Epoch 455/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.1567e-05 - mae: 0.0045 - val_loss: 9.0749e-04 - val_mae: 0.0311
    Epoch 456/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2200e-05 - mae: 0.0046 - val_loss: 6.1707e-04 - val_mae: 0.0240
    Epoch 457/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2811e-05 - mae: 0.0047 - val_loss: 8.1698e-04 - val_mae: 0.0284
    Epoch 458/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1906e-05 - mae: 0.0046 - val_loss: 8.5038e-04 - val_mae: 0.0292
    Epoch 459/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3723e-05 - mae: 0.0048 - val_loss: 0.0011 - val_mae: 0.0361
    Epoch 460/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.2042e-05 - mae: 0.0046 - val_loss: 6.4028e-04 - val_mae: 0.0247
    Epoch 461/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1353e-05 - mae: 0.0046 - val_loss: 8.0750e-04 - val_mae: 0.0279
    Epoch 462/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.0306e-05 - mae: 0.0044 - val_loss: 0.0011 - val_mae: 0.0351
    Epoch 463/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.3279e-05 - mae: 0.0047 - val_loss: 6.6622e-04 - val_mae: 0.0252
    Epoch 464/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.4082e-05 - mae: 0.0050 - val_loss: 6.8639e-04 - val_mae: 0.0251
    Epoch 465/500
    43/43 [==============================] - 1s 18ms/step - loss: 2.3005e-05 - mae: 0.0047 - val_loss: 0.0010 - val_mae: 0.0333
    Epoch 466/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1986e-05 - mae: 0.0046 - val_loss: 0.0011 - val_mae: 0.0360
    Epoch 467/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.1364e-05 - mae: 0.0046 - val_loss: 8.4336e-04 - val_mae: 0.0288
    Epoch 468/500
    43/43 [==============================] - 1s 19ms/step - loss: 2.0549e-05 - mae: 0.0044 - val_loss: 8.7372e-04 - val_mae: 0.0296
    




    <tensorflow.python.keras.callbacks.History at 0x1b6944643c8>




```python
model = keras.models.load_model("my_checkpoint")
```


```python
rnn_forecast = model_forecast(
    model,
    spy_normalized_to_traindata[x_test.index.min() - window_size:-1],
    window_size)[:, 0]
```


```python
rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()
rnn_unscaled_forecast.shape
```




    (427,)




```python
plt.figure(figsize=(10,6))

plt.title('SeqtoVec RNN Forecast')
plt.ylabel('Dollars $')
plt.xlabel('Timestep in Days')
plot_series(x_test.index, x_test)
plot_series(x_test.index, rnn_unscaled_forecast)
```


![png](/assets/images/RNN_seqtovec_seqtoseq/output_17_0.png)



```python
keras.metrics.mean_absolute_error(x_test, rnn_unscaled_forecast).numpy()
```




    24.051598



## Sequence-to-Sequence Forecasting


```python
def seq2seq_window_dataset(series, window_size, batch_size=128,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
```

The cell below illustrates exactly what the function above is doing. the cell above creates batches and laebls for those batches. The point Y[0] is the label for X[0] to try to calculate.


```python
# Show example of what seq-seq looks like
for X_batch, Y_batch in seq2seq_window_dataset(tf.range(10), 3, batch_size=1):
    print("X:", X_batch.numpy())
    print("Y:", Y_batch.numpy())
```

    X: [[[1]
      [2]
      [3]]]
    Y: [[[2]
      [3]
      [4]]]
    X: [[[2]
      [3]
      [4]]]
    Y: [[[3]
      [4]
      [5]]]
    X: [[[4]
      [5]
      [6]]]
    Y: [[[5]
      [6]
      [7]]]
    X: [[[3]
      [4]
      [5]]]
    Y: [[[4]
      [5]
      [6]]]
    X: [[[5]
      [6]
      [7]]]
    Y: [[[6]
      [7]
      [8]]]
    X: [[[6]
      [7]
      [8]]]
    Y: [[[7]
      [8]
      [9]]]
    X: [[[0]
      [1]
      [2]]]
    Y: [[[1]
      [2]
      [3]]]
    


```python
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30

# in the seq2seq_window we removed the need for the lambda layer to expand dimensions as it is already 3D 
train_set = seq2seq_window_dataset(normalized_x_train.flatten(), window_size,
                                   batch_size=128)

# Create model
model = keras.models.Sequential([
  keras.layers.SimpleRNN(100, return_sequences=True,
                         input_shape=[None, 1]),
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.Dense(1), # now dense layer is applied at every time step
])

lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-5 * 10**(epoch / 30))

# choose optimizer
optimizer = keras.optimizers.Nadam(lr=1e-5)

# compile model
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# create history callback from fit
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

    Epoch 1/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0577 - mae: 0.2484
    Epoch 2/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0261 - mae: 0.1691
    Epoch 3/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0147 - mae: 0.1251
    Epoch 4/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0093 - mae: 0.0957
    Epoch 5/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0067 - mae: 0.0767
    Epoch 6/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0053 - mae: 0.0647
    Epoch 7/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0045 - mae: 0.0575
    Epoch 8/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0039 - mae: 0.0520
    Epoch 9/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0035 - mae: 0.0475
    Epoch 10/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0031 - mae: 0.0435
    Epoch 11/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0027 - mae: 0.0400
    Epoch 12/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0024 - mae: 0.0370
    Epoch 13/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0021 - mae: 0.0343
    Epoch 14/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0019 - mae: 0.0318
    Epoch 15/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0017 - mae: 0.0296
    Epoch 16/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0015 - mae: 0.0276
    Epoch 17/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0013 - mae: 0.0256
    Epoch 18/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0011 - mae: 0.0237
    Epoch 19/100
    43/43 [==============================] - 1s 16ms/step - loss: 9.5482e-04 - mae: 0.0220
    Epoch 20/100
    43/43 [==============================] - 1s 16ms/step - loss: 8.2032e-04 - mae: 0.0203
    Epoch 21/100
    43/43 [==============================] - 1s 16ms/step - loss: 7.0278e-04 - mae: 0.0188
    Epoch 22/100
    43/43 [==============================] - 1s 16ms/step - loss: 5.9848e-04 - mae: 0.0174
    Epoch 23/100
    43/43 [==============================] - 1s 16ms/step - loss: 5.0908e-04 - mae: 0.0162
    Epoch 24/100
    43/43 [==============================] - 1s 16ms/step - loss: 4.3216e-04 - mae: 0.0152
    Epoch 25/100
    43/43 [==============================] - 1s 16ms/step - loss: 3.6795e-04 - mae: 0.0144
    Epoch 26/100
    43/43 [==============================] - 1s 15ms/step - loss: 3.3586e-04 - mae: 0.0148
    Epoch 27/100
    43/43 [==============================] - 1s 16ms/step - loss: 7.8029e-04 - mae: 0.0223
    Epoch 28/100
    43/43 [==============================] - 1s 17ms/step - loss: 2.8241e-04 - mae: 0.0146
    Epoch 29/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.9678e-04 - mae: 0.0117
    Epoch 30/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0017 - mae: 0.0346
    Epoch 31/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.5900e-04 - mae: 0.0114
    Epoch 32/100
    43/43 [==============================] - 1s 17ms/step - loss: 1.3248e-04 - mae: 0.0103
    Epoch 33/100
    43/43 [==============================] - 1s 17ms/step - loss: 1.1822e-04 - mae: 0.0099
    Epoch 34/100
    43/43 [==============================] - 1s 17ms/step - loss: 3.7125e-04 - mae: 0.0155
    Epoch 35/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0020 - mae: 0.0331
    Epoch 36/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.5567e-04 - mae: 0.0164
    Epoch 37/100
    43/43 [==============================] - 1s 16ms/step - loss: 9.9498e-05 - mae: 0.0096
    Epoch 38/100
    43/43 [==============================] - 1s 16ms/step - loss: 8.0396e-05 - mae: 0.0086
    Epoch 39/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0012 - mae: 0.0224
    Epoch 40/100
    43/43 [==============================] - 1s 16ms/step - loss: 4.3332e-04 - mae: 0.0196
    Epoch 41/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.2465e-04 - mae: 0.0115
    Epoch 42/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.7716e-04 - mae: 0.0133
    Epoch 43/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0017 - mae: 0.0305
    Epoch 44/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.8969e-04 - mae: 0.0125: 0s - loss: 2.1496e-04 - mae: 0.0
    Epoch 45/100
    43/43 [==============================] - 1s 16ms/step - loss: 5.8679e-05 - mae: 0.0077
    Epoch 46/100
    43/43 [==============================] - 1s 16ms/step - loss: 9.7122e-04 - mae: 0.0194
    Epoch 47/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.6680e-04 - mae: 0.0165
    Epoch 48/100
    43/43 [==============================] - 1s 17ms/step - loss: 8.3313e-05 - mae: 0.0093
    Epoch 49/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.5134e-04 - mae: 0.0103
    Epoch 50/100
    43/43 [==============================] - 1s 16ms/step - loss: 7.6863e-04 - mae: 0.0202
    Epoch 51/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.5445e-04 - mae: 0.0113
    Epoch 52/100
    43/43 [==============================] - 1s 16ms/step - loss: 4.8984e-05 - mae: 0.0071
    Epoch 53/100
    43/43 [==============================] - 1s 16ms/step - loss: 4.2433e-05 - mae: 0.0066
    Epoch 54/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0018 - mae: 0.0223
    Epoch 55/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0034 - mae: 0.0516
    Epoch 56/100
    43/43 [==============================] - 1s 16ms/step - loss: 9.7661e-05 - mae: 0.0100
    Epoch 57/100
    43/43 [==============================] - 1s 16ms/step - loss: 5.3855e-05 - mae: 0.0076
    Epoch 58/100
    43/43 [==============================] - 1s 16ms/step - loss: 4.4561e-05 - mae: 0.0069
    Epoch 59/100
    43/43 [==============================] - 1s 16ms/step - loss: 4.8780e-05 - mae: 0.0071
    Epoch 60/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.3181e-04 - mae: 0.0092
    Epoch 61/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.7572e-04 - mae: 0.0116
    Epoch 62/100
    43/43 [==============================] - 1s 17ms/step - loss: 5.9759e-05 - mae: 0.0078
    Epoch 63/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.4367e-04 - mae: 0.0107
    Epoch 64/100
    43/43 [==============================] - 1s 16ms/step - loss: 2.9736e-04 - mae: 0.0149
    Epoch 65/100
    43/43 [==============================] - 1s 16ms/step - loss: 3.5640e-04 - mae: 0.0164
    Epoch 66/100
    43/43 [==============================] - 1s 16ms/step - loss: 7.7354e-05 - mae: 0.0088
    Epoch 67/100
    43/43 [==============================] - 1s 16ms/step - loss: 1.8497e-04 - mae: 0.0101
    Epoch 68/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0498 - mae: 0.1164
    Epoch 69/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.1731 - mae: 0.4170
    Epoch 70/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0159 - mae: 0.1417
    Epoch 71/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0096 - mae: 0.0992
    Epoch 72/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0038 - mae: 0.0622
    Epoch 73/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0096 - mae: 0.0945
    Epoch 74/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0036 - mae: 0.0605
    Epoch 75/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0082 - mae: 0.0873
    Epoch 76/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0074 - mae: 0.0965
    Epoch 77/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0053 - mae: 0.0798
    Epoch 78/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0059 - mae: 0.0817
    Epoch 79/100
    43/43 [==============================] - 1s 18ms/step - loss: 0.0074 - mae: 0.0900
    Epoch 80/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0149 - mae: 0.1277
    Epoch 81/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0034 - mae: 0.0626
    Epoch 82/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0196 - mae: 0.1615
    Epoch 83/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0085 - mae: 0.0879
    Epoch 84/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0112 - mae: 0.1048
    Epoch 85/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0077 - mae: 0.0858
    Epoch 86/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0322 - mae: 0.2063
    Epoch 87/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0067 - mae: 0.0770
    Epoch 88/100
    43/43 [==============================] - 1s 18ms/step - loss: 0.0066 - mae: 0.0943
    Epoch 89/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0100 - mae: 0.1140
    Epoch 90/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.0124 - mae: 0.1282
    Epoch 91/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0108 - mae: 0.1196
    Epoch 92/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0224 - mae: 0.1672
    Epoch 93/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0151 - mae: 0.1322
    Epoch 94/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.2532 - mae: 0.5597
    Epoch 95/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0805 - mae: 0.2954
    Epoch 96/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0491 - mae: 0.2433
    Epoch 97/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.1068 - mae: 0.3529
    Epoch 98/100
    43/43 [==============================] - 1s 17ms/step - loss: 0.1202 - mae: 0.4006
    Epoch 99/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.1616 - mae: 0.4514
    Epoch 100/100
    43/43 [==============================] - 1s 16ms/step - loss: 0.0702 - mae: 0.2924
    


```python
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-5, 1, 0, .1])
```




    (1e-05, 1.0, 0.0, 0.1)




![png](/assets/images/RNN_seqtovec_seqtoseq/output_24_1.png)



```python
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# set window size and create input batch sequences
window_size = 20
train_set = seq2seq_window_dataset(normalized_x_train.flatten(), window_size,
                                   batch_size=128)
valid_set = seq2seq_window_dataset(normalized_x_valid.flatten(), window_size,
                                   batch_size=128)

# Create model for seq:seq RNN
model = keras.models.Sequential([
  keras.layers.SimpleRNN(100, return_sequences=True,
                         input_shape=[None, 1]),
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.Dense(1),
])

# choose optimizer and LR
optimizer = keras.optimizers.Nadam(lr=1e-3)

# set model params
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# set early stopping
early_stopping = keras.callbacks.EarlyStopping(patience=20)

# set model checkpoint to save best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)

# fit model
model.fit(train_set, epochs=500,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint])
```

    Epoch 1/500
         40/Unknown - 1s 15ms/step - loss: 0.0065 - mae: 0.0729INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 58ms/step - loss: 0.0072 - mae: 0.0782 - val_loss: 0.0058 - val_mae: 0.0863
    Epoch 2/500
    43/43 [==============================] - ETA: 0s - loss: 1.9874e-04 - mae: 0.0133INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 50ms/step - loss: 1.9874e-04 - mae: 0.0133 - val_loss: 0.0025 - val_mae: 0.0564
    Epoch 3/500
    43/43 [==============================] - 1s 14ms/step - loss: 2.4691e-04 - mae: 0.0117 - val_loss: 0.0065 - val_mae: 0.0836
    Epoch 4/500
    43/43 [==============================] - ETA: 0s - loss: 1.0095e-04 - mae: 0.0096INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 1.0095e-04 - mae: 0.0096 - val_loss: 0.0021 - val_mae: 0.0523
    Epoch 5/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.2069e-05 - mae: 0.0070 - val_loss: 0.0031 - val_mae: 0.0655
    Epoch 6/500
    43/43 [==============================] - 1s 15ms/step - loss: 1.3086e-04 - mae: 0.0088 - val_loss: 0.0253 - val_mae: 0.2082
    Epoch 7/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.8081e-04 - mae: 0.0141 - val_loss: 0.0184 - val_mae: 0.1800
    Epoch 8/500
    43/43 [==============================] - 1s 15ms/step - loss: 9.6989e-05 - mae: 0.0094 - val_loss: 0.0076 - val_mae: 0.1118
    Epoch 9/500
    38/43 [=========================>....] - ETA: 0s - loss: 3.4366e-05 - mae: 0.0059INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 49ms/step - loss: 4.5439e-05 - mae: 0.0067 - val_loss: 8.3669e-04 - val_mae: 0.0295
    Epoch 10/500
    43/43 [==============================] - 1s 15ms/step - loss: 1.3097e-04 - mae: 0.0092 - val_loss: 0.0026 - val_mae: 0.0650
    Epoch 11/500
    43/43 [==============================] - 1s 15ms/step - loss: 8.0387e-05 - mae: 0.0082 - val_loss: 0.0022 - val_mae: 0.0516
    Epoch 12/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.9197e-05 - mae: 0.0067 - val_loss: 9.1278e-04 - val_mae: 0.0371
    Epoch 13/500
    43/43 [==============================] - 1s 14ms/step - loss: 9.0905e-05 - mae: 0.0084 - val_loss: 0.0164 - val_mae: 0.1731
    Epoch 14/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.1635e-04 - mae: 0.0143 - val_loss: 0.0581 - val_mae: 0.3282
    Epoch 15/500
    43/43 [==============================] - 1s 16ms/step - loss: 0.0319 - mae: 0.0940 - val_loss: 0.2109 - val_mae: 0.6070
    Epoch 16/500
    43/43 [==============================] - 1s 14ms/step - loss: 0.0024 - mae: 0.0443 - val_loss: 0.0086 - val_mae: 0.1161
    Epoch 17/500
    43/43 [==============================] - 1s 14ms/step - loss: 2.7942e-04 - mae: 0.0143 - val_loss: 0.0025 - val_mae: 0.0589
    Epoch 18/500
    43/43 [==============================] - 1s 14ms/step - loss: 1.2714e-04 - mae: 0.0111 - val_loss: 0.0027 - val_mae: 0.0605
    Epoch 19/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.1412e-05 - mae: 0.0081 - val_loss: 0.0027 - val_mae: 0.0612
    Epoch 20/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.0783e-05 - mae: 0.0074 - val_loss: 0.0031 - val_mae: 0.0675
    Epoch 21/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.2293e-05 - mae: 0.0067 - val_loss: 0.0018 - val_mae: 0.0481
    Epoch 22/500
    43/43 [==============================] - 1s 15ms/step - loss: 8.2074e-05 - mae: 0.0082 - val_loss: 0.0088 - val_mae: 0.1216
    Epoch 23/500
    43/43 [==============================] - 1s 14ms/step - loss: 5.1582e-05 - mae: 0.0072 - val_loss: 0.0012 - val_mae: 0.0371
    Epoch 24/500
    43/43 [==============================] - 1s 14ms/step - loss: 5.6811e-05 - mae: 0.0073 - val_loss: 0.0048 - val_mae: 0.0880
    Epoch 25/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.4579e-05 - mae: 0.0059 - val_loss: 0.0018 - val_mae: 0.0486
    Epoch 26/500
    43/43 [==============================] - 1s 14ms/step - loss: 3.2853e-05 - mae: 0.0059 - val_loss: 0.0021 - val_mae: 0.0534
    Epoch 27/500
    38/43 [=========================>....] - ETA: 0s - loss: 2.9445e-05 - mae: 0.0055INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 54ms/step - loss: 4.9183e-05 - mae: 0.0066 - val_loss: 7.0121e-04 - val_mae: 0.0299
    Epoch 28/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.1044e-05 - mae: 0.0067 - val_loss: 0.0029 - val_mae: 0.0668
    Epoch 29/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.0450e-05 - mae: 0.0069 - val_loss: 0.0089 - val_mae: 0.1233
    Epoch 30/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.3613e-05 - mae: 0.0067 - val_loss: 0.0028 - val_mae: 0.0658
    Epoch 31/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.7468e-05 - mae: 0.0071 - val_loss: 0.0016 - val_mae: 0.0464
    Epoch 32/500
    43/43 [==============================] - 1s 15ms/step - loss: 5.9380e-05 - mae: 0.0078 - val_loss: 0.0039 - val_mae: 0.0794
    Epoch 33/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.7657e-05 - mae: 0.0066 - val_loss: 7.1066e-04 - val_mae: 0.0308
    Epoch 34/500
    38/43 [=========================>....] - ETA: 0s - loss: 3.5775e-05 - mae: 0.0062INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 50ms/step - loss: 4.2257e-05 - mae: 0.0067 - val_loss: 5.5704e-04 - val_mae: 0.0251
    Epoch 35/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.1218e-05 - mae: 0.0058 - val_loss: 0.0011 - val_mae: 0.0353
    Epoch 36/500
    43/43 [==============================] - 1s 14ms/step - loss: 7.0668e-05 - mae: 0.0072 - val_loss: 0.0023 - val_mae: 0.0577
    Epoch 37/500
    43/43 [==============================] - 1s 14ms/step - loss: 4.9520e-05 - mae: 0.0072 - val_loss: 0.0021 - val_mae: 0.0565
    Epoch 38/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.4139e-05 - mae: 0.0070 - val_loss: 0.0013 - val_mae: 0.0422
    Epoch 39/500
    43/43 [==============================] - 1s 14ms/step - loss: 6.7092e-05 - mae: 0.0080 - val_loss: 0.0060 - val_mae: 0.1014
    Epoch 40/500
    38/43 [=========================>....] - ETA: 0s - loss: 3.2352e-05 - mae: 0.0058INFO:tensorflow:Assets written to: my_checkpoint\assets
    43/43 [==============================] - 2s 53ms/step - loss: 4.3630e-05 - mae: 0.0067 - val_loss: 4.9787e-04 - val_mae: 0.0257
    Epoch 41/500
    43/43 [==============================] - 1s 14ms/step - loss: 3.7032e-05 - mae: 0.0063 - val_loss: 0.0026 - val_mae: 0.0644
    Epoch 42/500
    43/43 [==============================] - 1s 14ms/step - loss: 2.6789e-05 - mae: 0.0052 - val_loss: 0.0013 - val_mae: 0.0407
    Epoch 43/500
    43/43 [==============================] - 1s 14ms/step - loss: 1.1411e-04 - mae: 0.0081 - val_loss: 0.0034 - val_mae: 0.0722
    Epoch 44/500
    43/43 [==============================] - 1s 14ms/step - loss: 5.6307e-05 - mae: 0.0075 - val_loss: 7.1585e-04 - val_mae: 0.0278
    Epoch 45/500
    43/43 [==============================] - 1s 14ms/step - loss: 2.7848e-05 - mae: 0.0055 - val_loss: 0.0013 - val_mae: 0.0411
    Epoch 46/500
    43/43 [==============================] - 1s 14ms/step - loss: 3.2950e-05 - mae: 0.0057 - val_loss: 0.0032 - val_mae: 0.0723
    Epoch 47/500
    43/43 [==============================] - 1s 14ms/step - loss: 6.3031e-05 - mae: 0.0069 - val_loss: 0.0102 - val_mae: 0.1342
    Epoch 48/500
    43/43 [==============================] - 1s 14ms/step - loss: 5.0526e-05 - mae: 0.0073 - val_loss: 0.0025 - val_mae: 0.0632
    Epoch 49/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.7444e-05 - mae: 0.0064 - val_loss: 0.0054 - val_mae: 0.0967
    Epoch 50/500
    43/43 [==============================] - 1s 14ms/step - loss: 3.3599e-05 - mae: 0.0059 - val_loss: 5.4883e-04 - val_mae: 0.0240
    Epoch 51/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.6767e-05 - mae: 0.0053 - val_loss: 0.0011 - val_mae: 0.0385
    Epoch 52/500
    43/43 [==============================] - 1s 15ms/step - loss: 3.3276e-05 - mae: 0.0056 - val_loss: 0.0033 - val_mae: 0.0745
    Epoch 53/500
    43/43 [==============================] - 1s 15ms/step - loss: 2.8192e-05 - mae: 0.0053 - val_loss: 6.7243e-04 - val_mae: 0.0272
    Epoch 54/500
    43/43 [==============================] - 1s 15ms/step - loss: 4.5162e-05 - mae: 0.0062 - val_loss: 0.0013 - val_mae: 0.0434
    Epoch 55/500
    43/43 [==============================] - 1s 14ms/step - loss: 5.8065e-05 - mae: 0.0074 - val_loss: 0.0010 - val_mae: 0.0398
    Epoch 56/500
    43/43 [==============================] - 1s 14ms/step - loss: 6.2800e-05 - mae: 0.0076 - val_loss: 0.0019 - val_mae: 0.0547
    Epoch 57/500
    43/43 [==============================] - 1s 15ms/step - loss: 1.9001e-04 - mae: 0.0108 - val_loss: 0.0046 - val_mae: 0.0858
    Epoch 58/500
    43/43 [==============================] - 1s 15ms/step - loss: 6.1021e-05 - mae: 0.0077 - val_loss: 0.0012 - val_mae: 0.0403
    Epoch 59/500
    43/43 [==============================] - 1s 14ms/step - loss: 3.5384e-05 - mae: 0.0059 - val_loss: 0.0026 - val_mae: 0.0657
    Epoch 60/500
    43/43 [==============================] - 1s 16ms/step - loss: 2.6425e-05 - mae: 0.0052 - val_loss: 7.4362e-04 - val_mae: 0.0298
    




    <tensorflow.python.keras.callbacks.History at 0x1b69afcc948>




```python
# recall best model
model = keras.models.load_model("my_checkpoint")
```


```python
# create forecast and clip to only show test values
rnn_forecast = model_forecast(model, spy_normalized_to_traindata.flatten()[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[x_test.index.min() - window_size:-1, -1, 0]
```


```python
# Get data back to normal scale
rnn_unscaled_forecast = x_train_scaler.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()
rnn_unscaled_forecast.shape
```




    (427,)




```python
# Plot results
plt.figure(figsize=(10, 6))
plt.title('SeqtoSeq RNN')
plt.ylabel('Dollars $')
plt.xlabel('Timestep in Days')
plot_series(x_test.index, x_test)
plot_series(x_test.index, rnn_unscaled_forecast)
```


![png](/assets/images/RNN_seqtovec_seqtoseq/output_29_0.png)



```python
keras.metrics.mean_absolute_error(x_test, rnn_unscaled_forecast).numpy()
```




    4.3963356


