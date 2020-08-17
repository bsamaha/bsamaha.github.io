---
title: "Project: Kaggle Tweet Sentiment Extraction"
categories:
  - Kaggle
tags:
  - NLP
---

```python
from fastai.text import *
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

!pip install ipython-autotime
%load_ext autotime
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    

    Requirement already satisfied: ipython-autotime in /usr/local/lib/python3.6/dist-packages (0.1)
    




    (27480, 4) (3534, 3)
    time: 91.7 ms
    


```python
text_df = pd.concat([train_df,test_df]).drop(columns=['textID', 'selected_text'])
text_df
```