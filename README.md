# Text Classification with Multinomial Naive Bayes

1. Load Textual Data
2. Text Preprocessing (TF-IDF, word count)
3. Train Classifier
4. Evaluate Results
5. Test Model

## Importing Libraries


```python
from sklearn.naive_bayes import MultinomialNB # classifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # text vectorizer
#from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score  # evaluation
from sklearn.datasets import fetch_20newsgroups # data
import matplotlib.pyplot as plt # visualization
import pandas as pd # data representation
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
```

## 1. Load Textual Data

News articles in 20 different categories, for this tutorial we choose the following:
 - alt.atheism
 - comp.graphics
 - sci.med
 - soc.religion.christian


```python
news = fetch_20newsgroups()
```


```python
news.target_names
```




    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']




```python
target_categories = ['alt.atheism','comp.graphics','sci.med','soc.religion.christian']

train = fetch_20newsgroups(subset='train', categories=target_categories)
test = fetch_20newsgroups(subset='test', categories=target_categories)
```


```python
len(test.data), len(train.data)
```




    (1502, 2257)



### Sample


```python
print(f'CATEGORY: {target_categories[train.target[0]]}')
print('-' * 80)
print(train.data[0])
print('-' * 80)
```

    CATEGORY: comp.graphics
    --------------------------------------------------------------------------------
    From: sd345@city.ac.uk (Michael Collier)
    Subject: Converting images to HP LaserJet III?
    Nntp-Posting-Host: hampton
    Organization: The City University
    Lines: 14
    
    Does anyone know of a good way (standard PC application/PD utility) to
    convert tif/img/tga files into LaserJet III format.  We would also like to
    do the same, converting to HPGL (HP plotter) files.
    
    Please email any response.
    
    Is this the correct group?
    
    Thanks in advance.  Michael.
    -- 
    Michael Collier (Programmer)                 The Computer Unit,
    Email: M.P.Collier@uk.ac.city                The City University,
    Tel: 071 477-8000 x3769                      London,
    Fax: 071 477-8565                            EC1V 0HB.
    
    --------------------------------------------------------------------------------


## 2. Text preprocessing

Text must be represented as numbers (vectors). There are several useful techniques to transform text into vectors:
1. TF-IDF (Term Frequency - Inverse Document Frequency)
2. Word Count


```python
sample_sentences = [
    'My name is George, this is my name', 
    'I like apples', 
    'apple is my favorite fruit'
    ]
```

### TF-IDF


```python
tfidf = TfidfVectorizer()
```


```python
vectorizer = tfidf.fit_transform(sample_sentences)
```


```python
pd.DataFrame(vectorizer.toarray(), columns=tfidf.get_feature_names())
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
      <th>apple</th>
      <th>apples</th>
      <th>favorite</th>
      <th>fruit</th>
      <th>george</th>
      <th>is</th>
      <th>like</th>
      <th>my</th>
      <th>name</th>
      <th>this</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.306754</td>
      <td>0.466589</td>
      <td>0.000000</td>
      <td>0.466589</td>
      <td>0.613509</td>
      <td>0.306754</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.490479</td>
      <td>0.000000</td>
      <td>0.490479</td>
      <td>0.490479</td>
      <td>0.000000</td>
      <td>0.373022</td>
      <td>0.000000</td>
      <td>0.373022</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Words Counting


```python
count_vector = CountVectorizer()
```


```python
vectorizer = count_vector.fit_transform(sample_sentences)
```


```python
pd.DataFrame(vectorizer.toarray(), columns=count_vector.get_feature_names())
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
      <th>apple</th>
      <th>apples</th>
      <th>favorite</th>
      <th>fruit</th>
      <th>george</th>
      <th>is</th>
      <th>like</th>
      <th>my</th>
      <th>name</th>
      <th>this</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Model

Build two models, but use different vectorization techniques: TF-IDF and Word Count


```python
model_tfidf = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_count = make_pipeline(CountVectorizer(), MultinomialNB())
```

### 1. Training


```python
model_tfidf.fit(train.data, train.target), \
model_count.fit(train.data, train.target)
```




    (Pipeline(steps=[('tfidfvectorizer', TfidfVectorizer()),
                     ('multinomialnb', MultinomialNB())]),
     Pipeline(steps=[('countvectorizer', CountVectorizer()),
                     ('multinomialnb', MultinomialNB())]))



### 2. Predicting


```python
y_pred_tfidf = model_tfidf.predict(test.data)
y_pred_count = model_count.predict(test.data)
```

### 3. Evaluation


```python
f1 = f1_score(test.target, y_pred_tfidf, average='weighted')
accuracy = accuracy_score(test.target, y_pred_tfidf)
print('Multinomial Naive Bayes with TF-IDF:')
print('-' * 40)
print(f'f1: {f1:.4f}')
print(f'accuracy: {accuracy:.4f}')
```

    Multinomial Naive Bayes with TF-IDF:
    ----------------------------------------
    f1: 0.8368
    accuracy: 0.8349



```python
f1 = f1_score(test.target, y_pred_count, average='weighted')
accuracy = accuracy_score(test.target, y_pred_count)
print('Multinomial Naive Bayes with Word Count:')
print('-' * 40)
print(f'f1: {f1:.4f}')
print(f'accuracy: {accuracy:.4f}')
```

    Multinomial Naive Bayes with Word Count:
    ----------------------------------------
    f1: 0.9340
    accuracy: 0.9341


## 4. Testing the Model


```python
text = [
    'I believe in jesus', 
    'Nvidia released new video card', 
    'one apple a day takes a doctor away',
    'God does not exist',
    'My monitor supports HDR',
    'Vitamins are essential for your health and development'
]
```


```python
y_pred = model_tfidf.predict(text)
```


```python
for i in range(len(y_pred)):
    print(f'"{target_categories[y_pred[i]]:<22}" ==> "{text[i]}"')
```

    "soc.religion.christian" ==> "I believe in jesus"
    "comp.graphics         " ==> "Nvidia released new video card"
    "sci.med               " ==> "one apple a day takes a doctor away"
    "soc.religion.christian" ==> "God does not exist"
    "comp.graphics         " ==> "My monitor supports HDR"
    "sci.med               " ==> "Vitamins are essential for your health and development"



```python

```
