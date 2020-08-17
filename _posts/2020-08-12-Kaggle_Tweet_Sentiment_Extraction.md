---
title: "Project: Kaggle's Tweet Sentiment Extraction"
categories:
  - Project
  - Kaggle
  - NLP
tags:
  - Intrinsic Attention
  - Fastai
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
    


```python
# Import Data - Upload train.csv and test.csv files to Colab runtime folder
path = Path('/content')
train_df = pd.read_csv(f'{path}/train.csv')
test_df = pd.read_csv(f'{path}/test.csv')
submission_df = pd.read_csv(f'{path}/test.csv')
train_df.dropna(inplace=True)
print(train_df.shape, test_df.shape)
```

    (27480, 4) (3534, 3)
    time: 91.7 ms
    


```python
text_df = pd.concat([train_df,test_df]).drop(columns=['textID', 'selected_text'])
text_df
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
      <th>text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I`d have responded, if I were going</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sooo SAD I will miss you here in San Diego!!!</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>my boss is bullying me...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>what interview! leave me alone</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sons of ****, why couldn`t they put them on t...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3529</th>
      <td>its at 3 am, im very tired but i can`t sleep  ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3530</th>
      <td>All alone in this old house again.  Thanks for...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3531</th>
      <td>I know what you mean. My little dog is sinkin...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3532</th>
      <td>_sutra what is your next youtube video gonna b...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3533</th>
      <td>http://twitpic.com/4woj2 - omgssh  ang cute n...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
<p>31014 rows × 2 columns</p>
</div>



    time: 29.4 ms
    

# Data Exploration

What is the count of tweets in each sentiment class? Looking for possible class imbalances.


```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 27480 entries, 0 to 27480
    Data columns (total 4 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   textID         27480 non-null  object
     1   text           27480 non-null  object
     2   selected_text  27480 non-null  object
     3   sentiment      27480 non-null  object
    dtypes: object(4)
    memory usage: 1.0+ MB
    time: 7.88 ms
    


```python
train_df.isnull().sum()
```




    textID           0
    text             0
    selected_text    0
    sentiment        0
    dtype: int64



    time: 9.39 ms
    


```python
# Group sentiment classes together
temp = train_df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Purples')
```




<style  type="text/css" >
    #T_c4451628_dc99_11ea_85f6_0242ac1c0002row0_col1 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_c4451628_dc99_11ea_85f6_0242ac1c0002row1_col1 {
            background-color:  #dcdcec;
            color:  #000000;
        }    #T_c4451628_dc99_11ea_85f6_0242ac1c0002row2_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }</style><table id="T_c4451628_dc99_11ea_85f6_0242ac1c0002" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sentiment</th>        <th class="col_heading level0 col1" >text</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c4451628_dc99_11ea_85f6_0242ac1c0002level0_row0" class="row_heading level0 row0" >1</th>
                        <td id="T_c4451628_dc99_11ea_85f6_0242ac1c0002row0_col0" class="data row0 col0" >neutral</td>
                        <td id="T_c4451628_dc99_11ea_85f6_0242ac1c0002row0_col1" class="data row0 col1" >11117</td>
            </tr>
            <tr>
                        <th id="T_c4451628_dc99_11ea_85f6_0242ac1c0002level0_row1" class="row_heading level0 row1" >2</th>
                        <td id="T_c4451628_dc99_11ea_85f6_0242ac1c0002row1_col0" class="data row1 col0" >positive</td>
                        <td id="T_c4451628_dc99_11ea_85f6_0242ac1c0002row1_col1" class="data row1 col1" >8582</td>
            </tr>
            <tr>
                        <th id="T_c4451628_dc99_11ea_85f6_0242ac1c0002level0_row2" class="row_heading level0 row2" >0</th>
                        <td id="T_c4451628_dc99_11ea_85f6_0242ac1c0002row2_col0" class="data row2 col0" >negative</td>
                        <td id="T_c4451628_dc99_11ea_85f6_0242ac1c0002row2_col1" class="data row2 col1" >7781</td>
            </tr>
    </tbody></table>



    time: 39.8 ms
    


```python
# Group together sentiments
temp.style.background_gradient(cmap='Purples')

# Check ration of observations of each sentiment class -- looking for class imbalance
fig = go.Figure(go.Funnelarea(
    text =temp.sentiment,
    values = temp.text,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"},
    ))
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="9654ca19-d332-434a-a845-99c63e5459a3" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("9654ca19-d332-434a-a845-99c63e5459a3")) {
                    Plotly.newPlot(
                        '9654ca19-d332-434a-a845-99c63e5459a3',
                        [{"text": ["neutral", "positive", "negative"], "title": {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}, "type": "funnelarea", "values": [11117, 8582, 7781]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('9654ca19-d332-434a-a845-99c63e5459a3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


    time: 485 ms
    

### Question: What is the breakdown of words in "selected text" for each sentiment? 
We will use the jaccard similarity coefficient. This coefficient measure similarity for two *sets*. This number is between 0 and 1 with 0 being perfectly unique and 1 meaning the sets are identical.


```python
def jaccard(str1, str2): 
  """
  Inputs = 'text' column and 'selected_text' column 
  Output = % similarity between two sets
  """
  a = set(str1.lower().split()) 
  b = set(str2.lower().split())
  c = a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))
```

    time: 2.69 ms
    

#### Conclusion: Looking at the result of the code below we see that the neutral sentiment has a jaccard similarity score of 97% which means the text and selected_text columns for neutral sentiment are basically identical. We will use this information and will not be doing any text extraction on the neutral sentiment. We will simply copy the values from text into our selected text output.


```python
# Apply jaccard function to data frame 
train_df['jaccard_similarity'] = train_df.apply(lambda x: jaccard(x.text, x.selected_text), axis=1)

# Group by sentiment to see what the average jaccard similarity coefficient is
train_df.groupby('sentiment')['jaccard_similarity'].mean()
```




    sentiment
    negative    0.338613
    neutral     0.976445
    positive    0.314372
    Name: jaccard_similarity, dtype: float64



    time: 870 ms
    

#### Question: What is the jaccard score of tweets less than 3 words?

Is it possible to extract words out of very short tweets or should we just copy these values over as well?


```python
# Get number of words in selected text and add a new column to the df
train_df['num_words_selectedtext'] = train_df['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text

# Get number of words in text and add a new column to the df
train_df['Num_word_text'] = train_df['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text

# Plot distribution of jaccard scores for tweets less than 3 words
short_tweet_jaccard_score = train_df.loc[train_df['Num_word_text'] <=3]['jaccard_similarity']

# Plot distribution of jaccard score of tweets less than 3 words
short_tweet_jaccard_score.plot(kind = 'hist',title='Distribution of Jaccard Similarity for > 3 word tweets', bins = 5)

# Print out the median jaccard score
# median score = 1
print(f"Median jaccard similarity for tweets less than 3 words is {short_tweet_jaccard_score.median()}")
```

    Median jaccard similarity for tweets less than 3 words is 1.0
    


![png](output_14_1.png)


    time: 215 ms
    

#### Conclusion: The median jaccard similarity for these short tweets is a perfect 1. This means if we were looking to speed things up we could use any tweet that is less than 3 words and use that entire tweet for our selected_text output

## Word Clouds



```python
#install the appropriate library directly from their github and name the module for easy importing into google colab
!pip install git+https://github.com/amueller/word_cloud.git #egg=wordcloud
```

    Collecting git+https://github.com/amueller/word_cloud.git
      Cloning https://github.com/amueller/word_cloud.git to /tmp/pip-req-build-qxbrmz02
      Running command git clone -q https://github.com/amueller/word_cloud.git /tmp/pip-req-build-qxbrmz02
    Requirement already satisfied (use --upgrade to upgrade): wordcloud==1.6.0.post92+g51f9983 from git+https://github.com/amueller/word_cloud.git in /usr/local/lib/python3.6/dist-packages
    Requirement already satisfied: numpy>=1.6.1 in /usr/local/lib/python3.6/dist-packages (from wordcloud==1.6.0.post92+g51f9983) (1.18.5)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from wordcloud==1.6.0.post92+g51f9983) (7.0.0)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from wordcloud==1.6.0.post92+g51f9983) (3.2.2)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->wordcloud==1.6.0.post92+g51f9983) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->wordcloud==1.6.0.post92+g51f9983) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->wordcloud==1.6.0.post92+g51f9983) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->wordcloud==1.6.0.post92+g51f9983) (2.8.1)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from cycler>=0.10->matplotlib->wordcloud==1.6.0.post92+g51f9983) (1.15.0)
    Building wheels for collected packages: wordcloud
      Building wheel for wordcloud (setup.py) ... [?25l[?25hdone
      Created wheel for wordcloud: filename=wordcloud-1.6.0.post92+g51f9983-cp36-cp36m-linux_x86_64.whl size=338637 sha256=34d169fb34ced94e05b21859d5b30e9103acf2411d1e1af5549e71318a92c651
      Stored in directory: /tmp/pip-ephem-wheel-cache-s_wqlsjd/wheels/f6/e8/4a/d14000dae86311eb2b2f1adf9b179fc247858133c757ef42b3
    Successfully built wordcloud
    time: 13.5 s
    


```python
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

```

    time: 123 ms
    

#### Here we separate each sentiment into it's own dataframe to create word clouds for each sentiment


```python
pos_word_df = text_df[text_df['sentiment'] == 'positive']
# pos_word_df.head()
```

    time: 5.81 ms
    


```python
neg_word_df = text_df[text_df['sentiment'] == 'negative']
# neg_word_df.head()
```

    time: 4.48 ms
    


```python
neutral_word_df = text_df[text_df['sentiment'] == 'neutral']
# neutral_word_df.head()
```

    time: 4.56 ms
    


```python
# We create a string containing all words in the the tweets for each sentiment
all_text = " ".join(tweet for tweet in text_df.text)
pos_text = " ".join(tweet for tweet in pos_word_df.text)
neg_text = " ".join(tweet for tweet in neg_word_df.text)
neutral_text = " ".join(tweet for tweet in neutral_word_df.text)
```

    time: 14.8 ms
    


```python
# Here we use wordcloud's default list of stopwords
stopwords = set(STOPWORDS)
```

    time: 761 µs
    


```python
# Verifying our created string works in the wordcloud
worldcloud = WordCloud(stopwords=stopwords, background_color='white').generate(all_text)
plt.imshow(worldcloud, interpolation='bilinear')
```




    <matplotlib.image.AxesImage at 0x7f4f6f2f2b00>




![png](output_25_1.png)


    time: 1.72 s
    

#### Here we create the mask, over which our word clouds will be created


```python
#The image for which we will be using as our mask
Image.open('/content/twitter_mask.png')
```




![png](output_27_0.png)



    time: 104 ms
    


```python
# Import the twitter logo as an array and verify it's shape
twitter_mask = np.array(Image.open('/content/twitter_mask.png'))
twitter_mask.shape
```




    (1024, 1267, 4)



    time: 25.2 ms
    


```python
def transform_format(val):
  """
  Changes all values of 0 in the image array to 255 to ensure the background is white
  and will be excluded from the mask
  """
  for i in val:
    if i == 0:
      return 255
    else:
      return i
```

    time: 1.9 ms
    


```python
#We transform and verify the array of our image
transformed_twitter_mask = np.ndarray((twitter_mask.shape[0], twitter_mask.shape[1]), np.int32)

for i in range(len(twitter_mask)):
  transformed_twitter_mask[i] = list(map(transform_format, twitter_mask[i]))

transformed_twitter_mask
```




    array([[255, 255, 255, 255, ..., 255, 255, 255, 255],
           [255, 255, 255, 255, ..., 255, 255, 255, 255],
           [255, 255, 255, 255, ..., 255, 255, 255, 255],
           [255, 255, 255, 255, ..., 255, 255, 255, 255],
           ...,
           [255, 255, 255, 255, ..., 255, 255, 255, 255],
           [255, 255, 255, 255, ..., 255, 255, 255, 255],
           [255, 255, 255, 255, ..., 255, 255, 255, 255],
           [255, 255, 255, 255, ..., 255, 255, 255, 255]], dtype=int32)



    time: 2.56 s
    


```python
#Generate and show word cloud for all text
wc = WordCloud(background_color='white', mask=transformed_twitter_mask, stopwords=stopwords, contour_width=3, contour_color='dodgerblue', relative_scaling=.75)
wc.generate_from_text(all_text)
plt.figure(figsize = [30,20])
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```


![png](output_31_0.png)


    time: 4.26 s
    


```python
#Generate and show word cloud for all positive text
wc = WordCloud(background_color='white', mask=transformed_twitter_mask, stopwords=stopwords, contour_width=3, contour_color='dodgerblue')
wc.generate_from_text(pos_text)
plt.figure(figsize = [30,20])
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```


![png](output_32_0.png)


    time: 3.39 s
    


```python
#Generate and show word cloud for all negative text
wc = WordCloud(background_color='white', mask=transformed_twitter_mask, stopwords=stopwords, contour_width=3, contour_color='dodgerblue')
wc.generate(neg_text)
plt.figure(figsize = [30,20])
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```


![png](output_33_0.png)


    time: 3.53 s
    


```python
#Generate and show word cloud for all neutral text
wc = WordCloud(background_color='white', mask=transformed_twitter_mask, stopwords=stopwords, contour_width=3, contour_color='dodgerblue')
wc.generate(neutral_text)
plt.figure(figsize = [30,20])
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```


![png](output_34_0.png)


    time: 3.62 s
    

# Language Model

Building the language model we will use the ULMFit method as taught by Jeremy Howard at fastai. Since we are just building the language model the more text the better. We will combine all the tweets from the test and the train dfs. This will help our language model learn the tweet database dialect.  


```python
# Combine both train and test dataframes keeping "sentiment" and "text" columns
text_df = pd.concat([train_df,test_df]).drop(columns=['selected_text','textID'])

# Drop any null values
text_df.dropna(inplace=True)
```

    time: 22.6 ms
    


```python
# Use train_test_split to create a validation data set for the language model
from sklearn.model_selection import train_test_split

lm_train_df, lm_valid_df = train_test_split(text_df, test_size=.1)
```

    time: 147 ms
    


```python
# Create databunch class for fastai
data_lm = TextLMDataBunch.from_df(path='/content',train_df=lm_train_df,valid_df=lm_valid_df,text_cols='text', bs=64, max_vocab=100000, min_freq=2)
```









    time: 4.57 s
    


```python
# Show what the data bunch and see what it looks like after it has been cleaned and tokenized
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
      <td>who says i wasn`t just xxunk to myself ? xxbos happy early mothers day xxrep 4 ! xxbos xxmaj oh man , that`s rough . xxmaj sounded like the weekend went well ! xxmaj get some sleep xxbos i m the only ho that did nt go to xxunk tonite xxbos is watching xxup xxunk xxbos xxmaj math quiz : xxmaj if xxmaj meow = but xxmaj allergies = then</td>
    </tr>
    <tr>
      <td>1</td>
      <td>week meet me at xxup offf , next xxmaj xxunk , xxmaj fri and xxmaj sat . xxmaj and try there xxup mic and xxmaj xxunk . xxbos xxmaj no . xxmaj just sitting around xxunk tiny koi . xxbos i`m awake , too early for my liking on a sunday ... but i`m looking at my pictures last night was bloody awesome , there are no words ... xxbos</td>
    </tr>
    <tr>
      <td>2</td>
      <td>some tour dates stat ! xxbos i xxunk like to xxunk prawns , i also do nt like going shopping , running out of money and crawling round the car looking for more xxbos awake slept in a little cuz no construction . now studying bio a xxrep 5 l day but i do nt mind . xxbos i xxunk my shirt and i fail at screaming xxunk but i</td>
    </tr>
    <tr>
      <td>3</td>
      <td>hungry xxbos xxmaj love ' good girls go bad ' xxbos xxmaj my tummy hurts ... again xxbos i miss you xxrep 4 ! xxmaj it`s lonely and empty without you ! http : / / yfrog.com / xxunk xxbos damnit i didn`t but neither did you so win ! xxbos tomorrow night would definitely work xxbos _ peek xxunk : / did you tell him on msn ? xxunk</td>
    </tr>
    <tr>
      <td>4</td>
      <td>town trip . high for the day so far ? woke up to good ol ` fried chicken xxbos xxunk that sounds good , i hope you`re right xxbos happy mother`s day to all your mom`s xxbos xxmaj one taken from earlier this week ! http : / / twitpic.com / xxunk xxbos you are disappointing me xxrep 6 . xxbos xxmaj man with a great sense of humour ...</td>
    </tr>
  </tbody>
</table>


    time: 9.7 s
    


```python
# Create Learn object (same as instantiating a model in sklearn)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.7)
```

    Downloading https://s3.amazonaws.com/fast-ai-modelzoo/wt103-fwd.tgz
    





    time: 5.5 s
    


```python
# Find optimal learning rate hyper-parameter
learn.lr_find()
learn.recorder.plot()
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
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
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
      <progress value='99' class='' max='102' style='width:300px; height:20px; vertical-align: middle;'></progress>
      97.06% [99/102 00:10<00:00 10.9577]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_41_2.png)


    time: 12.4 s
    


```python
# Train the model
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
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
      <td>4.886467</td>
      <td>4.436565</td>
      <td>0.219717</td>
      <td>00:11</td>
    </tr>
  </tbody>
</table>


    time: 11.2 s
    


```python
# Train the model
learn.unfreeze()
learn.fit_one_cycle(4, 1e-3,moms=(0.8,0.7))
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
      <td>4.406765</td>
      <td>4.297631</td>
      <td>0.236514</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.269074</td>
      <td>4.216049</td>
      <td>0.246894</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.090111</td>
      <td>4.189057</td>
      <td>0.251711</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.972025</td>
      <td>4.191727</td>
      <td>0.250298</td>
      <td>00:14</td>
    </tr>
  </tbody>
</table>


    time: 56.6 s
    

## Test Language Model

Let's check and see if our language model puts together semi-coherent sentences


```python
text = 'I liked'
n_words = 10
n_sentences=3

print('\n'.join(learn.predict(text, n_words) for _ in range(n_sentences)))
```

    I liked that concert but disappointed i got that day xxbos
    I liked him so much . stole the eggs from his wife
    I liked contain his black version and now it won`t let me
    time: 585 ms
    


```python
# Save our language model and our encoder
learn.save('fine_tuned')
learn.save_encoder('fine_tuned_enc')
```

    time: 1.08 s
    

# Classifier

Now we will build and train our classifier


```python
# split the data for a validation data set
clas_train_df,clas_valid_df = train_test_split(train_df,test_size=0.2)
clas_train_df.shape,clas_valid_df.shape
```




    ((21984, 7), (5496, 7))



    time: 13.8 ms
    


```python
# Build data bunch for text classifier
data_clf = TextClasDataBunch.from_df(path = '/content', train_df = clas_train_df, valid_df = clas_valid_df, test_df=test_df, vocab=data_lm.vocab, text_cols='text', label_cols= 'sentiment', bs=64)
```













    time: 5.63 s
    


```python
# Create learn object for classifier
learn = text_classifier_learner(data_clf, arch = AWD_LSTM, drop_mult=0.6)
```

    time: 464 ms
    


```python
# Load encoder from the language model into our classifier
learn.load_encoder('fine_tuned_enc')

# Find best learning rate for our classifier
learn.lr_find()
learn.recorder.plot()
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
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
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
      <progress value='94' class='' max='343' style='width:300px; height:20px; vertical-align: middle;'></progress>
      27.41% [94/343 00:02<00:05 2.7257]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


![png](output_51_2.png)


    time: 2.99 s
    


```python
# Fit last layer
learn.fit_one_cycle(1, 2e-2)
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
      <td>0.864972</td>
      <td>0.781582</td>
      <td>0.644833</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>


    time: 6.9 s
    


```python
# Gradual Unfreeze
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-2), moms=(0.8,0.7))
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
      <td>0.805709</td>
      <td>0.700841</td>
      <td>0.691048</td>
      <td>00:08</td>
    </tr>
  </tbody>
</table>


    time: 8.39 s
    


```python
# Gradual unfreeze
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-2), moms=(0.8,0.7))
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
      <td>0.724487</td>
      <td>0.617378</td>
      <td>0.727256</td>
      <td>00:11</td>
    </tr>
  </tbody>
</table>


    time: 12 s
    


```python
# total unfreeze
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1e-2), moms=(0.8,0.7))
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
      <td>0.676735</td>
      <td>0.609669</td>
      <td>0.743814</td>
      <td>00:16</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.642357</td>
      <td>0.575969</td>
      <td>0.756550</td>
      <td>00:17</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.572286</td>
      <td>0.555733</td>
      <td>0.767649</td>
      <td>00:16</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.507962</td>
      <td>0.566313</td>
      <td>0.765284</td>
      <td>00:16</td>
    </tr>
  </tbody>
</table>


    time: 1min 6s
    

# Analyze Classifier Output


```python
# Save final model and reload
learn.save('final')
learn.load('final')

# view confusion matrix and build interp object
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
```






![png](output_57_1.png)


    time: 2.44 s
    


```python
# Text version of confusion matrix (helpful when you have a lot of classes)
interp.most_confused()
```




    [('negative', 'neutral', 348),
     ('neutral', 'negative', 284),
     ('neutral', 'positive', 278),
     ('positive', 'neutral', 257),
     ('positive', 'negative', 77),
     ('negative', 'positive', 46)]



    time: 182 ms
    


```python
# Sum up how many classifications our model got wrong
wrong = []
for idx in interp.most_confused():
  wrong.append(idx[2])
print(f'Total wrong sentiment classifications: {np.array(wrong).sum()}')
```

    Total wrong sentiment classifications: 1290
    time: 188 ms
    


```python
# Make predictions classifying sentiment on a string
learn.predict('this is very good!')
```




    (Category tensor(2), tensor(2), tensor([3.8357e-04, 7.3333e-04, 9.9888e-01]))



    time: 174 ms
    


```python
# Create interpretation object
text_interp = TextClassificationInterpretation.from_learner(learn)
```





    time: 1.19 s
    

# Selected Text Function



```python
import more_itertools as mit
import numpy as np

def list_output(intrinsic_attention_obj):
  """
  Takes in a fastaiv1 instrinsic attention obj and returns a list 
  of words and a list of the weights associated with those words
  """
  word_str = str(intrinsic_attention_obj[0])
  word_list = word_str.split()
  weight_list = intrinsic_attention_obj[1]
  return word_list, weight_list

def list_filter(word_list, weight_list, vocab):
  """
  Takes in a list of words and a list of weights then returns filtered versions
  based upon the provided vocabulary

  A filtered or sliced version of the vocab can be passed to filter out unwanted
  tokens or words
  """
  filtered_words = []
  filtered_weights = []
  for word, weight in zip(word_list, weight_list):
    if word in vocab:
      filtered_words.append(word)
      filtered_weights.append(float(weight))

  return filtered_words, filtered_weights

def attention_weight_grouper(filtered_weights, threshold):
  """
  Takes in a list of weights and a provided threshold. Returns a list of the indexes
  of the weights that exceed this threshold. 
  """
  weights = []
  ind_list = []
  idx = 0
  for weight in filtered_weights:
    if weight >= threshold:
      weights.append(weight)
      ind_list.append(idx)
    idx += 1
  return  ind_list

def text_output(ind_list, filtered_words, filtered_weights):
  """
  Takes in and ind_list outputed from the attention_weight_grouper function
  along with its filtered words and filtered weights lists. 

  Checks the ind_list for consecutive indices and groups them. Takes the sum of all
  weights associated with the indices in the group and stores the group of indices
  with the largest weight value.

  Returns a string containing the words associated with the indices stored
  """
  groupings = [list(group) for group in mit.consecutive_groups(ind_list)]
  grouping_sum = 0
  grouping_ind = 0
  idx = 0
  for group in groupings:
    if len(group) > 1:
      total = 0
      for id in group:
        total += filtered_weights[id]
        if total > grouping_sum:
          grouping_sum = total
          grouping_ind = idx
      idx += 1
    else:
      ref = group[0]
      total = filtered_weights[ref]
      if total > grouping_sum:
          grouping_sum = total
          grouping_ind = idx
      idx += 1

  #create an empty string and append the chosen words with an extra space at the end
  #to give the text the appropriate format when outputting multiple words
  selected_text = ''
  if len(groupings) < 1:
    np.nan
  if len(groupings) >= 1:
    for id in groupings[grouping_ind]:
      selected_text += f'{filtered_words[id]} '
  else:
    try:
      if len(groupings[0]) == 1:
          selected_text = f'{filtered_words} '
      if len(groupings[0]) > 1:
        for id in groupings[0]:
          selected_text += f'{filtered_words[id]} '
      else:
        ref = groupings[0][0]
        selected_text += f'{filtered_words[ref]} '
    except:
      np.nan
  #remove the extra space from the end of string string
  selected_text = selected_text[:-1]
  return selected_text

def selected_text_grabber(intrinsic_attention_obj, vocab = data_lm.vocab.itos[9:], threshold = .35):
  """
  Takes in a fastai intrinsic attention obj, a filtered/sliced vocab, and a threshold for significance
  determiniation and applies the list_output, list_filter, attention_weight_grouper, and text_output
  functions to them to transform the object into a string of 'selected_text' 
  """
  word, wt = list_output(intrinsic_attention_obj)
  fil_word, fil_wt = list_filter(word, wt, vocab)
  atn_ind = attention_weight_grouper(fil_wt, threshold)
  return text_output(atn_ind, fil_word, fil_wt)
```

    time: 53.6 ms
    

# Outputs
This outputs section is for preparing the data to be submitted in the format that Kaggle is looking for.



```python
submission_df.head()
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
      <th>textID</th>
      <th>text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f87dea47db</td>
      <td>Last session of the day  http://twitpic.com/67ezh</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96d74cb729</td>
      <td>Shanghai is also really exciting (precisely -...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>eee518ae67</td>
      <td>Recession hit Veronique Branquinho, she has to...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01082688c6</td>
      <td>happy bday!</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33987a8ee5</td>
      <td>http://twitpic.com/4w75p - I like it!!</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



    time: 12.3 ms
    

### Selected Text Grabber Function Test


```python
def df_filter(df, match, text_grabber):
  """
  This function checks the data frame if the sentiment is neutral or not. If it is
  neutral the data will be copied directly from the text column. If the sentiment
  is not neutral it will use the text_grabber function to select the important
  text.
  """
  if df['sentiment'] == match:
    return df['text']
  return text_grabber(text_interp.intrinsic_attention(df['text']))
```

    time: 1.81 ms
    


```python
# Apply text grabber function on our data frame to prepare it for submission
submission_df['selected_text'] = submission_df.apply(lambda x: df_filter(x, 'neutral', selected_text_grabber), axis = 1)

submission_df.head()
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
      <th>textID</th>
      <th>text</th>
      <th>sentiment</th>
      <th>selected_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f87dea47db</td>
      <td>Last session of the day  http://twitpic.com/67ezh</td>
      <td>neutral</td>
      <td>Last session of the day  http://twitpic.com/67ezh</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96d74cb729</td>
      <td>Shanghai is also really exciting (precisely -...</td>
      <td>positive</td>
      <td>exciting</td>
    </tr>
    <tr>
      <th>2</th>
      <td>eee518ae67</td>
      <td>Recession hit Veronique Branquinho, she has to...</td>
      <td>negative</td>
      <td>shame</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01082688c6</td>
      <td>happy bday!</td>
      <td>positive</td>
      <td>happy bday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33987a8ee5</td>
      <td>http://twitpic.com/4w75p - I like it!!</td>
      <td>positive</td>
      <td>like</td>
    </tr>
  </tbody>
</table>
</div>



    time: 5min 41s
    

### Visuals of Intrinsic Attention

---




*   The darker the red colors means that word's weight is closer to 0 and the word provides negligible value to the sentiment of the sentence
*   The darker the green color means that word's weight is closer to 1 signifying it holds a lot of importance in the regards of its sentiment classification




```python
submission_df[submission_df.isnull().any(axis=1)]
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
      <th>textID</th>
      <th>text</th>
      <th>sentiment</th>
      <th>selected_text</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



    time: 7.76 ms
    


```python
# Example of the desired selected text we hope to achieve 
train_df.iloc[1]
```


```python
# Show what intrinsic attention looks like
text_interp.show_intrinsic_attention(train_df['text'][1])
```


```python
# Demonstrate what the selected text of our function looks like
selected_text_grabber(text_interp.intrinsic_attention(train_df['text'][1]))
```


```python
# Show what the raw output looks like that we changed into our "selected_text"
text_interp.intrinsic_attention(train_df['text'][1])
```


```python
sample_df = test_df.head()
sample_df['selected_text'] = sample_df.apply(lambda x: df_filter(x, 'neutral', selected_text_grabber), axis = 1)
sample_df
```


```python

```
