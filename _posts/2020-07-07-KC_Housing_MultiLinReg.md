---
title: "King County Multivariate Linear Regression Project"
categories:
  - Project
tags:
  - Machine Learning - Linear Regression
---


# [View the full GitHub Repo](https://github.com/bsamaha/KC_Housing_Multivariate_Regression)

# Import Libraries & Data for Model


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math
import seaborn as sns
from my_func import *
import pylab

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

%matplotlib inline
```


```python
df = pd.read_csv('../CSV Files/linreg_ready_data.csv')
display(df.shape)
display(df.columns)
```


    (20110, 71)



    Index(['log_price', 'log_sqft_living', 'zipcode_98002', 'zipcode_98003',
           'zipcode_98004', 'zipcode_98005', 'zipcode_98006', 'zipcode_98007',
           'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014',
           'zipcode_98019', 'zipcode_98022', 'zipcode_98023', 'zipcode_98024',
           'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030',
           'zipcode_98031', 'zipcode_98032', 'zipcode_98033', 'zipcode_98034',
           'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042',
           'zipcode_98045', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055',
           'zipcode_98056', 'zipcode_98058', 'zipcode_98059', 'zipcode_98065',
           'zipcode_98070', 'zipcode_98072', 'zipcode_98074', 'zipcode_98075',
           'zipcode_98077', 'zipcode_98092', 'zipcode_98102', 'zipcode_98103',
           'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108',
           'zipcode_98109', 'zipcode_98112', 'zipcode_98115', 'zipcode_98116',
           'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122',
           'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136',
           'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98155',
           'zipcode_98166', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178',
           'zipcode_98188', 'zipcode_98198', 'zipcode_98199'],
          dtype='object')


***
## Assumptions of OLS Model

Each assumption if violated means we may have to take extra steps to improve our model or in some cases dump the model altogether. Here is a list of the assumptions of the model:

- A linear relationship is assumed between the dependent variable and the independent variables.
- Regression residuals must be normally distributed.
- The residuals are homoscedastic and approximately rectangular-shaped.
- Absence of multicollinearity is expected in the model, meaning that independent variables are not too highly correlated.
- No Autocorrelation of the residuals.

I will be explaining these assumptions in more detail as we arrive at each of them in the tutorial. At this point, however, we need to have an idea of what they are.

# Build The Model

#### Create Test : Train Split


```python
#print(values_to_be_dropped)
#after running the model delete the hash to see what columns need to be dropped from dataframe

df = df.drop(columns=['zipcode_98023','zipcode_98032'], axis=1)
```


```python
# define our input variable (X) & output variable
X = df.drop(columns=['log_price'], axis=1)
Y = df[['log_price']]

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

# create a Linear Regression model object
regression_model = LinearRegression()

# pass through the X_train & y_train data set
regression_model.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



#### Explore The Output


```python
# let's grab the coefficient of our model and the intercept
intercept = regression_model.intercept_[0]
coefficent = float(regression_model.coef_[0][0])

print("The intercept for our model is {:}".format(intercept))
print('-'*100)

# loop through the dictionary and print the data
for coef in zip(X.columns, regression_model.coef_[0]):
    print("The Coefficient for {} is {:.3}".format(coef[0],coef[1]))
```

    The intercept for our model is 7.432651901502438
    ----------------------------------------------------------------------------------------------------
    The Coefficient for log_sqft_living is 0.674
    The Coefficient for zipcode_98002 is -0.0425
    The Coefficient for zipcode_98003 is 0.0331
    The Coefficient for zipcode_98004 is 1.18
    The Coefficient for zipcode_98005 is 0.84
    The Coefficient for zipcode_98006 is 0.759
    The Coefficient for zipcode_98007 is 0.711
    The Coefficient for zipcode_98008 is 0.714
    The Coefficient for zipcode_98010 is 0.23
    The Coefficient for zipcode_98011 is 0.469
    The Coefficient for zipcode_98014 is 0.346
    The Coefficient for zipcode_98019 is 0.327
    The Coefficient for zipcode_98022 is 0.0784
    The Coefficient for zipcode_98024 is 0.462
    The Coefficient for zipcode_98027 is 0.594
    The Coefficient for zipcode_98028 is 0.447
    The Coefficient for zipcode_98029 is 0.663
    The Coefficient for zipcode_98030 is 0.0515
    The Coefficient for zipcode_98031 is 0.0862
    The Coefficient for zipcode_98033 is 0.85
    The Coefficient for zipcode_98034 is 0.545
    The Coefficient for zipcode_98038 is 0.177
    The Coefficient for zipcode_98039 is 1.4
    The Coefficient for zipcode_98040 is 1.04
    The Coefficient for zipcode_98042 is 0.0838
    The Coefficient for zipcode_98045 is 0.38
    The Coefficient for zipcode_98052 is 0.7
    The Coefficient for zipcode_98053 is 0.66
    The Coefficient for zipcode_98055 is 0.126
    The Coefficient for zipcode_98056 is 0.341
    The Coefficient for zipcode_98058 is 0.183
    The Coefficient for zipcode_98059 is 0.367
    The Coefficient for zipcode_98065 is 0.439
    The Coefficient for zipcode_98070 is 0.479
    The Coefficient for zipcode_98072 is 0.559
    The Coefficient for zipcode_98074 is 0.658
    The Coefficient for zipcode_98075 is 0.702
    The Coefficient for zipcode_98077 is 0.582
    The Coefficient for zipcode_98092 is 0.0435
    The Coefficient for zipcode_98102 is 1.04
    The Coefficient for zipcode_98103 is 0.849
    The Coefficient for zipcode_98105 is 0.988
    The Coefficient for zipcode_98106 is 0.33
    The Coefficient for zipcode_98107 is 0.87
    The Coefficient for zipcode_98108 is 0.35
    The Coefficient for zipcode_98109 is 1.03
    The Coefficient for zipcode_98112 is 1.09
    The Coefficient for zipcode_98115 is 0.833
    The Coefficient for zipcode_98116 is 0.817
    The Coefficient for zipcode_98117 is 0.834
    The Coefficient for zipcode_98118 is 0.471
    The Coefficient for zipcode_98119 is 1.02
    The Coefficient for zipcode_98122 is 0.844
    The Coefficient for zipcode_98125 is 0.557
    The Coefficient for zipcode_98126 is 0.574
    The Coefficient for zipcode_98133 is 0.462
    The Coefficient for zipcode_98136 is 0.729
    The Coefficient for zipcode_98144 is 0.688
    The Coefficient for zipcode_98146 is 0.282
    The Coefficient for zipcode_98148 is 0.155
    The Coefficient for zipcode_98155 is 0.444
    The Coefficient for zipcode_98166 is 0.358
    The Coefficient for zipcode_98168 is 0.0582
    The Coefficient for zipcode_98177 is 0.669
    The Coefficient for zipcode_98178 is 0.14
    The Coefficient for zipcode_98188 is 0.0746
    The Coefficient for zipcode_98198 is 0.0996
    The Coefficient for zipcode_98199 is 0.922
    

# Evaluate the Model


```python
# define our intput
X2 = sm.add_constant(X)

# create a OLS model
model = sm.OLS(Y, X2,hasconst=True)

# fit the data
est = model.fit()

# check the data
print(est.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:              log_price   R-squared:                       0.816
    Model:                            OLS   Adj. R-squared:                  0.815
    Method:                 Least Squares   F-statistic:                     1306.
    Date:                Fri, 05 Jun 2020   Prob (F-statistic):               0.00
    Time:                        09:52:50   Log-Likelihood:                 3668.8
    No. Observations:               20110   AIC:                            -7200.
    Df Residuals:                   20041   BIC:                            -6654.
    Df Model:                          68                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const               7.4397      0.031    241.929      0.000       7.379       7.500
    log_sqft_living     0.6728      0.004    167.719      0.000       0.665       0.681
    zipcode_98002      -0.0407      0.016     -2.569      0.010      -0.072      -0.010
    zipcode_98003       0.0498      0.014      3.604      0.000       0.023       0.077
    zipcode_98004       1.1755      0.015     77.846      0.000       1.146       1.205
    zipcode_98005       0.8418      0.018     47.003      0.000       0.807       0.877
    zipcode_98006       0.7639      0.012     63.896      0.000       0.740       0.787
    zipcode_98007       0.7093      0.019     38.218      0.000       0.673       0.746
    zipcode_98008       0.6942      0.014     49.939      0.000       0.667       0.721
    zipcode_98010       0.2670      0.023     11.363      0.000       0.221       0.313
    zipcode_98011       0.4751      0.016     29.690      0.000       0.444       0.507
    zipcode_98014       0.3341      0.023     14.782      0.000       0.290       0.378
    zipcode_98019       0.3264      0.017     19.183      0.000       0.293       0.360
    zipcode_98022       0.0790      0.016      4.843      0.000       0.047       0.111
    zipcode_98024       0.4516      0.030     14.971      0.000       0.393       0.511
    zipcode_98027       0.5943      0.013     47.379      0.000       0.570       0.619
    zipcode_98028       0.4408      0.014     31.971      0.000       0.414       0.468
    zipcode_98029       0.6629      0.013     50.302      0.000       0.637       0.689
    zipcode_98030       0.0505      0.014      3.531      0.000       0.022       0.079
    zipcode_98031       0.0830      0.014      5.957      0.000       0.056       0.110
    zipcode_98033       0.8450      0.012     70.382      0.000       0.821       0.869
    zipcode_98034       0.5477      0.011     49.947      0.000       0.526       0.569
    zipcode_98038       0.1697      0.011     15.635      0.000       0.148       0.191
    zipcode_98039       1.3774      0.049     27.872      0.000       1.281       1.474
    zipcode_98040       1.0360      0.015     67.997      0.000       1.006       1.066
    zipcode_98042       0.0850      0.011      7.758      0.000       0.063       0.106
    zipcode_98045       0.3727      0.016     23.309      0.000       0.341       0.404
    zipcode_98052       0.7004      0.011     65.060      0.000       0.679       0.722
    zipcode_98053       0.6618      0.013     51.762      0.000       0.637       0.687
    zipcode_98055       0.1438      0.014     10.238      0.000       0.116       0.171
    zipcode_98056       0.3458      0.012     28.761      0.000       0.322       0.369
    zipcode_98058       0.1833      0.012     15.736      0.000       0.160       0.206
    zipcode_98059       0.3707      0.012     31.710      0.000       0.348       0.394
    zipcode_98065       0.4356      0.014     31.739      0.000       0.409       0.463
    zipcode_98070       0.4844      0.027     18.027      0.000       0.432       0.537
    zipcode_98072       0.5607      0.014     39.519      0.000       0.533       0.589
    zipcode_98074       0.6564      0.012     54.846      0.000       0.633       0.680
    zipcode_98075       0.7019      0.013     53.239      0.000       0.676       0.728
    zipcode_98077       0.5983      0.017     34.795      0.000       0.565       0.632
    zipcode_98092       0.0489      0.013      3.676      0.000       0.023       0.075
    zipcode_98102       1.0270      0.022     47.522      0.000       0.985       1.069
    zipcode_98103       0.8429      0.011     79.861      0.000       0.822       0.864
    zipcode_98105       0.9876      0.015     63.731      0.000       0.957       1.018
    zipcode_98106       0.3295      0.013     25.487      0.000       0.304       0.355
    zipcode_98107       0.8682      0.014     61.630      0.000       0.841       0.896
    zipcode_98108       0.3354      0.016     20.686      0.000       0.304       0.367
    zipcode_98109       1.0424      0.021     49.584      0.000       1.001       1.084
    zipcode_98112       1.0722      0.015     71.311      0.000       1.043       1.102
    zipcode_98115       0.8292      0.011     78.144      0.000       0.808       0.850
    zipcode_98116       0.8185      0.013     62.618      0.000       0.793       0.844
    zipcode_98117       0.8316      0.011     76.877      0.000       0.810       0.853
    zipcode_98118       0.4618      0.011     41.186      0.000       0.440       0.484
    zipcode_98119       1.0277      0.017     61.804      0.000       0.995       1.060
    zipcode_98122       0.8465      0.014     61.491      0.000       0.820       0.874
    zipcode_98125       0.5681      0.012     47.215      0.000       0.545       0.592
    zipcode_98126       0.5758      0.013     45.678      0.000       0.551       0.601
    zipcode_98133       0.4652      0.011     41.446      0.000       0.443       0.487
    zipcode_98136       0.7348      0.014     51.655      0.000       0.707       0.763
    zipcode_98144       0.6781      0.013     52.421      0.000       0.653       0.703
    zipcode_98146       0.2772      0.014     20.143      0.000       0.250       0.304
    zipcode_98148       0.1435      0.028      5.125      0.000       0.089       0.198
    zipcode_98155       0.4437      0.012     37.889      0.000       0.421       0.467
    zipcode_98166       0.3590      0.015     24.444      0.000       0.330       0.388
    zipcode_98168       0.0503      0.014      3.587      0.000       0.023       0.078
    zipcode_98177       0.6657      0.015     45.312      0.000       0.637       0.695
    zipcode_98178       0.1250      0.014      8.731      0.000       0.097       0.153
    zipcode_98188       0.0850      0.019      4.565      0.000       0.049       0.121
    zipcode_98198       0.0956      0.014      6.844      0.000       0.068       0.123
    zipcode_98199       0.9174      0.013     68.475      0.000       0.891       0.944
    ==============================================================================
    Omnibus:                      842.307   Durbin-Watson:                   1.994
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2752.096
    Skew:                          -0.057   Prob(JB):                         0.00
    Kurtosis:                       4.809   Cond. No.                         301.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
(abs(df.corr().log_price).sort_values(ascending=False) > .1).sum()
```




    25



## Hypothesis Testing

With hypothesis testing, we are trying to determine the statistical significance of the coefficient estimates. This test is outlined as the following.

- **Null Hypothesis:** There is no relationship between the exploratory variables and the dependent variable.
- **Alternative Hypothesis:** There is a relationship between the exploratory variables and the dependent variable.
***
- If we **reject the null**, we are saying there is a relationship, and the coefficients do not equal 0.
- If we **fail to reject the null**, we are saying there is no relationship, and the coefficients do equal 0
***

## Assumption 1
- A linear relationship is assumed between the dependent variable and the independent variables.


```python
#return to top of notebook and remove these columns
values_to_be_dropped = est.pvalues.loc[est.pvalues>.05]
est.pvalues.loc[est.pvalues>=.05]
```




    Series([], dtype: float64)




```python
sns.distplot(est.pvalues)
plt.title("Parameter Pvalue Distribution")
```




    Text(0.5, 1.0, 'Parameter Pvalue Distribution')




![Param_P-Value_Distribution](/assets/images/KC_MultiLinReg/Param_P-Value_Distribution.png)


**If there are no values present in the output above 0.05 we reject the null hypothesis and all varaiables in our model are significant.**



## Assumption 2
### Normally Distributed Residuals
- Regression residuals must be normally distributed.


```python
# check for the normality of the residuals
sm.qqplot(est.resid, line='s')
pylab.show()

# also check that the mean of the residuals is approx. 0.
mean_residuals = sum(est.resid)/ len(est.resid)
print("The mean of the residuals is {:.4}".format(mean_residuals))
```


![png](/assets/images/KC_MultiLinReg/QQ_Plot.png)


    The mean of the residuals is 1.253e-14
    

**This is not an airtight test as we can see the residuals on the tails break away from the line, but to me this does not look terrible as it is not until 2 deviations where it begins to take off. The mean of the residuals is still near 0.**

## Assumption 3
- The residuals are homoscedastic and approximately rectangular-shaped.

### White's Test for heteroscedasticity


```python
# Run the White's test
_, pval, __, f_pval = diag.het_white(est.resid, est.model.exog)
print("Pval:",pval,"************ F_pval:", f_pval)
print('-'*100)

# print the results of the test
if pval > 0.05:
    print("For the White's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")
    
else:
    print("For the White's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is heterosecdasticity. \n")


```
    Pval: 5.203411872121607e-266, F_pval: 7.825766394899378e-280
    ----------------------------------------------------------------------------------------------------
    For the White's Test
    The p-value was 5.203e-266
    We reject the null hypthoesis, so there is heterosecdasticity. 

### Breusch-Pagan's Test for Homoscedasticity


```python
# Run the Breusch-Pagan test
_, pval, __, f_pval = diag.het_breuschpagan(est.resid, est.model.exog)
print("Pval:",pval,"************ F_pval:", f_pval)
print('-'*100)

# print the results of the test
if pval > 0.05:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is Homoskedasticity.")

else:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is no Homoskedasticity.")
```

    Pval: 4.144787067358851e-198, F_pval: 8.676129540258215e-205
    ----------------------------------------------------------------------------------------------------
    For the Breusch-Pagan's Test
    The p-value was 4.145e-198
    We reject the null hypthoesis, so there is no Homoskedasticity.

## Assumption 4
### Multicolinearality


```python
# the VIF does expect a constant term in the data, so we need to add one using the add_constant method
X3 = sm.tools.add_constant(X)

# create the series for both
vif = pd.Series([variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])], index=X3.columns)

#checking for values greater than 3
display(vif.loc[vif >= 3])


```


    const    466.215579
    dtype: float64


A general recommendation is that if any of our variables come back with a value of 5 or higher, then they should be removed from the model. I decided to show you how the VFI comes out before we drop the highly correlated variables and after we remove the highly correlated variables.

**If no values (besides constant) greater than 5 there is no multicolinearality present**


## Assumption 5 
### Autocorrelation


### Durbin-Watson Test for Autocorrelation
***
### How to test for autocorrelation?
We will go to the statsmodels.stats.diagnostic module, and use the Ljung-Box test for no autocorrelation of residuals. Here:

- **H0: The data are random.**
- **Ha: The data are not random.**

That means we want to fail to reject the null hypothesis, have a large p-value because then it means we have no autocorrelation. To use the Ljung-Box test, we will call the `acorr_ljungbox` function, pass through the `est.resid` and then define the lags. 

The lags can either be calculated by the function itself, or we can calculate them. If the function handles it the max lag will be `min((num_obs // 2 - 2), 40)`, however, there is a rule of thumb that for non-seasonal time series the lag is ` min(10, (num_obs // 5))`.

We also can visually check for autocorrelation by using the `statsmodels.graphics` module to plot a graph of the autocorrelation factor.


```python
# calculate the lag, optional
lag = min(10, (len(X)//5))
#lag2 = min((len(X) // 2 - 2),40)
print('The number of lags will be {}'.format(lag))
print('-'*100)

# run the Ljung-Box test for no autocorrelation of residuals
test_results = diag.acorr_ljungbox(est.resid, lags = lag)

# grab the p-values and the test statistics
ibvalue, p_val = test_results

# print the results of the test
if min(p_val) > 0.05:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("We fail to reject the null hypthoesis, so there is no autocorrelation.")
    print('-'*100)
else:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("We reject the null hypthoesis, so there is autocorrelation.")
    print('-'*100)

# plot autocorrelation
sm.graphics.tsa.plot_pacf(est.resid,lags=lag)
plt.show()
```

    The number of lags will be 10
    ----------------------------------------------------------------------------------------------------
    The lowest p-value found was 0.4288
    We fail to reject the null hypthoesis, so there is no autocorrelation.
    ----------------------------------------------------------------------------------------------------
    

    C:\Users\blake\Anaconda3\lib\site-packages\statsmodels\stats\diagnostic.py:524: FutureWarning: The value returned will change to a single DataFrame after 0.12 is released.  Set return_df to True to use to return a DataFrame now.  Set return_df to False to silence this warning.
      warnings.warn(msg, FutureWarning)
    


![Autocorrelation](/assets/images/KC_MultiLinReg/autocorrelation.png)


### Model Accuracy Test


```python
cv_results  = cross_val_score(regression_model, X_test, y_test, cv=5)
print("Accuracy: %0.3f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))
```

    Accuracy: 0.817 (+/- 0.03)
    

### Make Predictions to Test the Model


```python
# Get multiple predictions
y_predict = regression_model.predict(X_test)

# Show the first 5 predictions
y_dollars = np.e**y_predict
y_dollars[:5].round(2)
```




    array([[383836.44],
           [312834.87],
           [640324.52],
           [289162.23],
           [600954.6 ]])




```python
# calculate the mean squared error
model_mse = mean_squared_error(np.e**y_test, np.e**y_predict)

# calculate the mean absolute error
model_mae = mean_absolute_error(np.e**y_test, np.e**y_predict)

# calulcate the root mean squared error
model_rmse =  math.sqrt(model_mse)

# display the output
print(f"MSE = ${round(model_mse,2)} ")

print(f"MAE = ${round(model_mae,2)}")

print(f"RMSE = ${round(model_rmse,2)}")
```

    MSE = $11715575217.32 
    MAE = $73233.34
    RMSE = $108238.51
    

# Save for Later Use


```python
import pickle

# pickle the model.
with open('kc_county_multilinreg.sav','wb') as f:
     pickle.dump(regression_model,f)

# load it back in to see if everything works
with open('kc_county_multilinreg.sav', 'rb') as pickle_file:
     kc_county_houseprice_predictor = pickle.load(pickle_file)
```

### Bring back in to see if regression still works


```python
print(X_test.iloc[1])
print('*'*100)
print('*'*100)
print('These are the data points:',X_test.iloc[1].loc[X_test.iloc[1] > 1])

```

    log_sqft_living    7.700748
    zipcode_98002      0.000000
    zipcode_98003      1.000000
    zipcode_98004      0.000000
    zipcode_98005      0.000000
                         ...   
    zipcode_98177      0.000000
    zipcode_98178      0.000000
    zipcode_98188      0.000000
    zipcode_98198      0.000000
    zipcode_98199      0.000000
    Name: 9515, Length: 68, dtype: float64
    ****************************************************************************************************
    ****************************************************************************************************
    These are the data points: log_sqft_living    7.700748
    Name: 9515, dtype: float64
    


```python
kc_county_houseprice_predictor.predict(np.array(X_test[:1]))

```




    array([[12.8579718]])




```python
#function located in my_func.py to convert log output to readable dollars
log_to_dollars(kc_county_houseprice_predictor.predict(X_test[:1]))
```




    array([[383836.44]])




```python
kc_county_houseprice_predictor.predict(np.array(X_test[46:47]))
```




    array([[12.55640263]])




```python
from my_func import *

#output test was a numpy array so just did this so you can see a dollar value
np.asscalar(log_to_dollars(kc_county_houseprice_predictor.predict(X_test[46:47])))
```
    283907.18






