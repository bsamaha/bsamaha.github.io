---
title: "Project: Kaggle Disaster Tweet Classification"
categories:
  - Kaggle
  - Project
tags:
  - NLP
  - Deep Learning
  - Fastai
---

This was a hastily done project as it was my first foray into NLP with Fastai. I built this notebook in Google Colab as I do with most deep learning projects. This notebook must should be run in an environment with a GPU.

```python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.text import * 
from pathlib import Path

```


```python
# Set up path for training data
train_path = '/content/drive/My Drive/Colab Notebooks/Kaggle - Disaster Tweet Classification/train.csv'
train_df = pd.read_csv(train_path)
```


```python
# Set up path for test data
test_path = '/content/drive/My Drive/Colab Notebooks/Kaggle - Disaster Tweet Classification/test.csv'
test_df = pd.read_csv(test_path)
```


```python
# Finish setting up data for project
base_path="./output"
text_columns=['text']
label_columns=['target']
BATCH_SIZE=128
```


```python
# View text
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Concat all data and test 'text' to help build a better language model
tweets = pd.concat([train_df[text_columns], test_df[text_columns]])
print(tweets.shape)
```

    (10876, 1)
    


```python
# Create Language model databunch
data_lm = (TextList.from_df(tweets)
           #Inputs: all the text files in path
            .split_by_rand_pct(0.10)
           #We randomly split and keep 10% for validation
            .label_for_lm()          
           #We want to do a language model so we label accordingly
            .databunch(bs=BATCH_SIZE))

# Save language model
data_lm.save('tmp_lm')
```










```python
# View language data with all preprocessing done
data_lm.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>idx</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>orders in xxmaj california xxbos xxmaj just got sent this photo from xxmaj xxunk # xxmaj alaska as smoke from # wildfires xxunk into a school xxbos # rockyfire xxmaj update = &gt; xxmaj california xxmaj hwy . 20 closed in both xxunk due to xxmaj lake xxmaj county fire - # xxunk # wildfires xxbos # flood # disaster xxmaj heavy rain causes flash flooding of streets in xxmaj</td>
    </tr>
    <tr>
      <td>1</td>
      <td>xxunk xxbos ' xxmaj the man who can drive himself further once the effort gets xxunk is the man who will win . ' \n  xxmaj xxunk xxmaj xxunk xxbos 320 [ xxup ir ] xxup icemoon [ xxup aftershock ] | http : / / t.co / xxunk | @djicemoon | # xxmaj dubstep # trapmusic # dnb # xxup edm # xxmaj dance # icesû _ http</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ambulance helicopter crash http : / / t.co / xxunk xxbos xxunk waiting for an ambulance xxbos @fouseytube you ok ? xxmaj need a ambulance . xxmaj xxunk that was good ! http : / / t.co / xxunk xxbos xxup ambulance xxup sprinter xxup automatic xxup frontline xxup vehicle xxup choice xxup of 14 xxup lez xxup compliant | ebay http : / / t.co / xxunk xxbos xxmaj</td>
    </tr>
    <tr>
      <td>3</td>
      <td>the xxmaj salt xxmaj river xxmaj wild xxmaj horse ... https : / / t.co / xxunk via @change xxbos xxmaj please sign &amp; &amp; xxup rt to save # xxunk http : / / t.co / xxunk http : / / t.co / xxunk xxbos xxmaj world xxmaj annihilation vs xxmaj self xxmaj transformation http : / / t.co / xxunk xxmaj xxunk xxmaj attack to xxmaj xxunk xxmaj</td>
    </tr>
    <tr>
      <td>4</td>
      <td>http : / / t.co / o91f3cyy0r xxunk xxbos xxmaj one xxmaj direction xxmaj is my pick for http : / / t.co / q2eblokeve xxmaj fan xxmaj army # xxmaj directioners http : / / t.co / encmhz6y34 xxunk xxbos 5 xxmaj seconds of xxmaj summer xxmaj is my pick for http : / / t.co / xxunk xxmaj fan xxmaj army # xxup 5sosfam http : / /</td>
    </tr>
  </tbody>
</table>



```python
# Create language model learner object to train
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
```


```python
# View details of language model
print('Model Summary:')
print(learn.layer_groups)
```

    Model Summary:
    [Sequential(
      (0): WeightDropout(
        (module): LSTM(400, 1152, batch_first=True)
      )
      (1): RNNDropout()
    ), Sequential(
      (0): WeightDropout(
        (module): LSTM(1152, 1152, batch_first=True)
      )
      (1): RNNDropout()
    ), Sequential(
      (0): WeightDropout(
        (module): LSTM(1152, 400, batch_first=True)
      )
      (1): RNNDropout()
    ), Sequential(
      (0): Embedding(5432, 400, padding_idx=1)
      (1): EmbeddingDropout(
        (emb): Embedding(5432, 400, padding_idx=1)
      )
      (2): LinearDecoder(
        (decoder): Linear(in_features=400, out_features=5432, bias=True)
        (output_dp): RNNDropout()
      )
    )]
    


```python
# Find Learning Training Rate
learn.lr_find()

learn.recorder.plot(suggestion=True)
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='3' class='' max='4' style='width:300px; height:20px; vertical-align: middle;'></progress>
      75.00% [3/4 00:13<00:04]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.632857</td>
      <td>#na#</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.570141</td>
      <td>#na#</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.835607</td>
      <td>#na#</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='12' class='' max='29' style='width:300px; height:20px; vertical-align: middle;'></progress>
      41.38% [12/29 00:01<00:02 8.3502]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    Min numerical gradient: 5.75E-02
    Min loss divided by 10: 6.31E-02
    


![png](/assets/images/Kaggle_Tweet_Disaster_Classification/output_10_2.png)



```python
# Relaod and Train Language Model More
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.473547</td>
      <td>2.950933</td>
      <td>0.477679</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.445255</td>
      <td>2.934130</td>
      <td>0.482227</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.377565</td>
      <td>2.886792</td>
      <td>0.488644</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.271737</td>
      <td>2.882216</td>
      <td>0.495201</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.156252</td>
      <td>2.899906</td>
      <td>0.494894</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.050642</td>
      <td>2.923301</td>
      <td>0.495731</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.960514</td>
      <td>2.945938</td>
      <td>0.496456</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.861020</td>
      <td>2.957307</td>
      <td>0.497628</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.785011</td>
      <td>2.970374</td>
      <td>0.496931</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.744761</td>
      <td>2.966264</td>
      <td>0.497461</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>



```python
# Save encoder and language model
learn.save_encoder('lm_enc')
learn.save('lm_fitted')
```


```python
Build a Text Classifer 
```

# Build a Text Classifer 


```python
# Create data bunch for classifier
data_clas = (TextList.from_df(train_df, cols=text_columns, vocab=data_lm.vocab)
             .split_by_rand_pct(0.1)
             .label_from_df('target')
             .add_test(test_df[text_columns])
             .databunch(bs=BATCH_SIZE))

# Save databunch for classifier
data_clas.save('tmp_clas')
```














```python
# Create Text Classifier Object
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
```


```python
# Load encoder into classifier
learn.load_encoder('lm_fit')
```


```python
# Gradual unfreeze method as taught by fastai
learn.freeze_to(-1)
learn.summary()
```


```python
# Find Best Training rate for classifier object
learn.lr_find()
learn.recorder.plot(suggestion=True)
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='1' class='' max='2' style='width:300px; height:20px; vertical-align: middle;'></progress>
      50.00% [1/2 00:02<00:02]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.721497</td>
      <td>#na#</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='40' class='' max='53' style='width:300px; height:20px; vertical-align: middle;'></progress>
      75.47% [40/53 00:01<00:00 1.2437]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    Min numerical gradient: 2.09E-03
    Min loss divided by 10: 5.25E-02
    


![png](/assets/images/Kaggle_Tweet_Disaster_Classification/output_19_2.png)



```python
# Train last layer of classifier
learn.fit_one_cycle(10, slice(2e-3,1e-1))

```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.514742</td>
      <td>0.444491</td>
      <td>0.809461</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.514687</td>
      <td>0.449156</td>
      <td>0.802891</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.538397</td>
      <td>0.488367</td>
      <td>0.795007</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.550148</td>
      <td>0.516797</td>
      <td>0.777924</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.567057</td>
      <td>0.447139</td>
      <td>0.797635</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.546917</td>
      <td>0.478468</td>
      <td>0.780552</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.536333</td>
      <td>0.490575</td>
      <td>0.781866</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.519825</td>
      <td>0.430909</td>
      <td>0.818660</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.517218</td>
      <td>0.430311</td>
      <td>0.809461</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.511057</td>
      <td>0.430546</td>
      <td>0.813403</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>



```python
learn.save('classifier-stage1')
```


```python
learn.load('classifier-stage1')

```


```python
# Unfreeze all layers and train again
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-6, 1e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.505692</td>
      <td>0.435985</td>
      <td>0.810775</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.488305</td>
      <td>0.423120</td>
      <td>0.812089</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.490391</td>
      <td>0.423960</td>
      <td>0.817346</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.479405</td>
      <td>0.428408</td>
      <td>0.812089</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.472055</td>
      <td>0.418933</td>
      <td>0.816032</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>



```python
# Export model to pkl file
learn.export()
```


```python
# Save model
learn.save('final_classifier')
```


```python
# Look at results of model
interp = TextClassificationInterpretation.from_learner(learn)
interp.show_top_losses(10)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Text</th>
      <th>Prediction</th>
      <th>Actual</th>
      <th>Loss</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>xxbos # xxup xxunk : xxmaj fire xxmaj truck xxmaj service xxmaj body for xxup xxunk ( xxmaj fire fighting rescue &amp; &amp; safety equipment xxmaj xxunk ... http : / / t.co / xxunk</td>
      <td>1</td>
      <td>0</td>
      <td>3.21</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>xxbos xxmaj need to stop xxunk things up because when everything eventually explodes the casualties just keep getting higher and higher</td>
      <td>0</td>
      <td>1</td>
      <td>3.10</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>xxbos xxmaj calgary news weather and traffic for xxmaj august 5 * ~ 90 http : / / t.co / xxunk http : / / t.co / xxunk</td>
      <td>1</td>
      <td>0</td>
      <td>3.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>xxbos all that panicking made me tired ; _ _ ; i want to sleep in my bed</td>
      <td>0</td>
      <td>1</td>
      <td>3.01</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>xxbos xxup cdc has a pretty cool list of all bioterrorism xxunk xxunk</td>
      <td>0</td>
      <td>1</td>
      <td>2.92</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>xxbos a traumatised dog that was found buried up to its head in dirt in xxmaj france is now in safe hands . xxmaj this is such a ... http : / / t.co / xxunk</td>
      <td>1</td>
      <td>0</td>
      <td>2.82</td>
      <td>0.06</td>
    </tr>
    <tr>
      <td>xxbos xxmaj if you 're in search of powerful content to xxunk your business or have been xxunk with the deluge of ' xxunk : / / t.co / xxunk</td>
      <td>0</td>
      <td>1</td>
      <td>2.63</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>xxbos xxunk xxunk xxmaj oh okay i just got the message twice and got xxunk . xxmaj sorry . i 'll check it out !</td>
      <td>0</td>
      <td>1</td>
      <td>2.62</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>xxbos xxmaj someone teaching you that xxunk will obliterate trials in your life is trying to sell you a used car . xxmaj jesus 's life blows that theory . '</td>
      <td>0</td>
      <td>1</td>
      <td>2.62</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>xxbos xxunk funny cause my dumb ass was the young one to get n trouble the most lol</td>
      <td>0</td>
      <td>1</td>
      <td>2.53</td>
      <td>0.08</td>
    </tr>
  </tbody>
</table>



```python
# Create and View Confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
```






![png](/assets/images/Kaggle_Tweet_Disaster_Classification/output_27_1.png)


# Predictions


```python
# View what output of predictions look like
learn.predict(test_df.loc[0,'text'])
```




    (Category tensor(1), tensor(1), tensor([0.3700, 0.6300]))






```python
def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in learn.data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]
```


```python
test_preds = get_preds_as_nparray(DatasetType.Test)
```






```python
# Code for formatting for Kaggle Submission
sample_submission = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Kaggle - Disaster Tweet Classification/sample_submission.csv')
sample_submission['target'] = np.argmax(test_preds, axis=1)
sample_submission.to_csv('/content/drive/My Drive/Colab Notebooks/Kaggle - Disaster Tweet Classification/submissions.csv', index=False, header=True)
```


```python
# Check ratio of target variable to see if it is similar to the original training distribution
sample_submission['target'].value_counts()
```




    0    2080
    1    1183
    Name: target, dtype: int64


