---
title: "Project: DrivenData's Pump it Up"
categories:
  - Project
  - DrivenData
tags:
  - Multilabel Classification
  - XGBoost
---
Please view this brief presentation I put together to see a rough map of how I went about mining insights from this data. I used the distributions of the status group to look for interesting ratios in different features. Meaning, if the normal ratio was 1:1:1 that means for every "functional" well, there is a "funcational needs repair", and a "non functional well." I dissected the features and looked for data where the ratios were different, like 2:2:1 for example. It is with these unique ratios the XGBClassifier is able to discern the differences in what makes a well functional or not.

[Presentation PDF](https://github.com/bsamaha/Competition---DrivenData---Pump-It-Up/blob/master/Presentation.pdf)

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost
```


```python
df = pd.read_csv('./train_test_data/prepared_training_data.csv',index_col='id')

test_df = pd.read_csv('./train_test_data/prepared_test_data.csv',index_col='id')
#df.drop(columns='permit',inplace=True)
#test_df.drop(columns='permit',inplace=True)

```


```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier, plot_importance, plot_tree

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,plot_confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFECV
```


```python
# set X and y variables
X = df.drop(columns=['status_group','population'])
Y = df.status_group

# get list of predictors to OHE
object_predictors = list(X.select_dtypes(include=['object']).columns)
test_object_predictors = list(test_df.select_dtypes(include=['object']).columns)
```


```python
# Check Columns are good
X.columns
```




    Index(['date_recorded', 'gps_height', 'longitude', 'latitude', 'region',
           'district_code', 'scheme_management', 'permit', 'extraction_type',
           'management', 'quality_group', 'quantity', 'source_type',
           'waterpoint_type', 'tsh_bins', 'top_funders', 'top_installers',
           'population_size', 'construction_decade'],
          dtype='object')




```python
# Check columns are good
test_df.columns
```




    Index(['date_recorded', 'gps_height', 'longitude', 'latitude', 'region',
           'district_code', 'scheme_management', 'permit', 'extraction_type',
           'management', 'quality_group', 'quantity', 'source_type',
           'waterpoint_type', 'tsh_bins', 'top_funders', 'top_installers',
           'population_size', 'construction_decade'],
          dtype='object')




```python
 # Make Column transformer for object columns and ignore the rest
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), object_predictors),
    remainder='passthrough')

 # Make Column transformer for object columns and ignore the rest
test_column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), test_object_predictors),
    remainder='passthrough')

# Fit column transformers
X = column_trans.fit_transform(X)
test_data = column_trans.transform(test_df)
```


```python
print(X.shape)

print(test_data.shape)
```

    (59400, 122)
    (14850, 122)
    

### Train Test Split


```python
# Train test split for model training data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=.2,
    random_state=0)
```


```python
# tune the model
n_estimators_1 = [200,300]
max_depth_1 = [15,20]
learning_rate_1 = [0.1]
min_child_weight_1 = [1]
reg_lamba_1 = [20,25]
subsample_1 = [0.75,1]

# Create dictionary of parameters to search
param_grid = dict(max_depth=max_depth_1,
                  n_estimators=n_estimators_1,
                  learning_rate=learning_rate_1,
                  min_child_weight=min_child_weight_1,
                  reg_lambda=reg_lamba_1,
                  subsample=subsample_1,
                 )
```


```python
# Cross Validate and gridsearch
clf = XGBClassifier(objective='multi:softmax',early_stopping_rounds=10)
kfold = StratifiedKFold(n_splits=3,shuffle=True,random_state=1)
grid_search = GridSearchCV(clf,param_grid,scoring='accuracy',n_jobs=-1,cv=kfold,verbose=1)
grid_result = grid_search.fit(X, Y)
```

    Fitting 3 folds for each of 16 candidates, totalling 48 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 24 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  48 out of  48 | elapsed:  5.5min finished
    


```python
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

    Best: 0.810320 using {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 25, 'subsample': 0.75}
    0.809529 (0.000999) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 20, 'subsample': 0.75}
    0.808620 (0.001044) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 20, 'subsample': 1}
    0.810320 (0.001022) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 25, 'subsample': 0.75}
    0.808519 (0.001390) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 25, 'subsample': 1}
    0.807710 (0.001248) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 20, 'subsample': 0.75}
    0.809057 (0.001055) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 20, 'subsample': 1}
    0.808502 (0.001197) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 25, 'subsample': 0.75}
    0.808805 (0.000889) with: {'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 25, 'subsample': 1}
    0.808535 (0.001606) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 20, 'subsample': 0.75}
    0.808047 (0.001029) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 20, 'subsample': 1}
    0.809040 (0.001278) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 25, 'subsample': 0.75}
    0.808569 (0.000866) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 25, 'subsample': 1}
    0.805337 (0.001452) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 20, 'subsample': 0.75}
    0.806633 (0.001068) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 20, 'subsample': 1}
    0.806229 (0.000707) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 25, 'subsample': 0.75}
    0.807340 (0.000683) with: {'learning_rate': 0.1, 'max_depth': 20, 'min_child_weight': 1, 'n_estimators': 300, 'reg_lambda': 25, 'subsample': 1}
    


```python
# Plot Confusion matrix to see how best estimator does
plot_confusion_matrix(grid_result.best_estimator_, X, Y)
plt.xticks(rotation=45)
plt.figure(figsize=(10,10))
```




    <Figure size 720x720 with 0 Axes>




![png](/assets/images/DrivenData_Pump_it_up/DrivenData_image.png)


    <Figure size 720x720 with 0 Axes>



```python
# Make predictions to be submitted
test_preds = grid_result.best_estimator_.predict(test_data)
```

#### Send Test to CSV to submit


```python
# View normalized breakdown of data to check ratios
pd.Series(test_preds).value_counts(normalize=True)*100
```




    functional                 59.973064
    non functional             35.993266
    functional needs repair     4.033670
    dtype: float64




```python
# Check DF
test_df.reset_index(inplace=True)
test_df['status_group'] = test_preds
submit_df = test_df[['id','status_group']]
```


```python
submit_df.head()
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
      <th>status_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50785</td>
      <td>non functional</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51630</td>
      <td>functional</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17168</td>
      <td>functional</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45559</td>
      <td>non functional</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49871</td>
      <td>functional</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame.to_csv(submit_df,path_or_buf='./Predictions/iteration_7_csv',index=False)
```


```python
import pickle

# pickle the model.
with open(f'./{grid_result.best_score_:.4}_iteration_5.sav','wb') as f:
     pickle.dump(clf,f)


```


```python
import pickle
#load it back in to see if everything works
with open('0.8005330203672462_iteration_3.sav', 'rb') as pickle_file:
     clf = pickle.load(pickle_file)
```


```python
clf.predict(test_data)
```
