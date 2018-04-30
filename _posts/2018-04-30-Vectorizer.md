---
layout: post
title: Vectorizer
comments: true
tags: [ml, vectorizer]
category: ml
---

*본문에서는 machine leraning 분야에 사용되는 Vectorizer에 대해 다룹니다.*

Vectorizer는 말 그대로 어떤 대상을 vector로 바꾸는 작업을 수행하는 개체를 의미한다. 기계학습 분야에서는 주로 텍스트를 쉽게 분석하기 위해 벡터로 표현할 때 사용한다.

숫자나 벡터를 인풋으로 기대하는 여러 머신러닝 모델을 실행하기 위해서는 텍스트나 또 다른 형태의 데이터를 숫자나 벡터로 나타낼 필요가 있고,  이 때 사용하는 것이 바로 `vectorizer`이다 . 현재 Scikit-learn 패키지에서 여러 vectorizor를 제공하고 있으며, 이에 대해 살펴보려고 한다. 

## BOG(Bag Of Word)
문서를 벡터로 변환하는 가장 기본적인 방법으로, 문서 내의 모든 단어를 모아 가방(Bag of words)를 만들고, 특정 문서에 어떤 단어가 들어있는지를 리스트 형태로 순서-숫자로 나타내는 것이다.

예를들어 다음과 같이 두 문장이 있다고 가정하자.

	sentence1 = 'This is a black cat'
	sentence2 = 'This is a white dog'

이 문장(문서)들에 들어있는 단어의 리스트는 다음과 같다.

	BOG(Bag Of Word) = ['this', 'is', 'a', 'black', 'cat', 'white', 'dog']
	
우리는 각 단어에 고유 번호를 줄 수 있다. 이것은 우리에게 일종의 사전(단어의 모음)이 된다. 

	dictionary = {'a': 2, 'black': 3, 'cat': 4, 'dog': 6, 'is': 1, 'this': 0, 'white': 5}

위에서 입력받은 sentence1, sentence2는 tokenize를 통해 각 단어로 나뉘고, 사전을 거치며 n번째 단어가 문서에 포함되어있는지 확인 후 0, 1로 포함여부를 나타낸다.(vectorize)

	# Tokenize
	sentence1 = ['This', 'is', 'a', 'black', 'cat']
	sentence2 = ['This', 'is', 'a', 'white', 'dog']

	# vectorize
	# dictionary: ['this', 'is', 'a', 'black', 'cat', 'white', 'dog']
	sentence1 = [1, 1, 1, 1, 1, 0, 0]
	sentence2 = [1, 1, 1, 0, 0, 1, 1]

Scikit-Learn에서는 여러가지 vectorizer를 제공하는데 이번 포스트에서는 `CountVectorizer`, `TfidfVectorizer`에 대해 자세히 알아보자.

## CountVectorizer
`CountVectorizer`는 문서 집합(문서 리스트)에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW 기반으로 벡터를 만든다. 여기에서는 주로 두가지 함수를 이용한다.

1. Tokenize(문서나 문장을 토큰으로 변환)(`fit()`)
2.  각 문서에서 토큰의 출현 빈도 카운트 및 BOW 인코딩 벡터로 변환(`transform()`)



```python
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```


```python
corpus = [
    'This is a black cat',
    'This is the white dog.',
    'Is your dog black?',
    'My favorite colour is white',
    'A black cat wears a white hat',    
]
vect.fit(corpus)
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)




```python
vect.vocabulary_
```




    {'black': 0,
     'cat': 1,
     'colour': 2,
     'dog': 3,
     'favorite': 4,
     'hat': 5,
     'is': 6,
     'my': 7,
     'the': 8,
     'this': 9,
     'wears': 10,
     'white': 11,
     'your': 12}




```python
vect.transform(['This is a black dog with a black hat']).toarray()
```




    array([[2, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]], dtype=int64)




```python
vect.transform(['I really want to do that']).toarray()
```




    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)


## TfidfVectorizer
`TfidfVectorizer`는  TF-IDF 방식으로 단어의 가중치를 조정한 BOW 벡터를 만든다.

### TFIDF
- TF(Term Frequency): 특정 단어의 빈도수
- DF(Document Frequency): 특정 단어가 들어가있는 문서의 수
- IDF: DF의 역수
$$ TfIdf = tf \times idf$$

TFIDF를 이용하면 많은 문서에 등장하는 단어는 비중이 작아지고, 특정 문서군에서만 등장하는 단어는 비중이 높아진다.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidv = TfidfVectorizer().fit(corpus)
tfidv.transform(corpus).toarray()
```




    array([[ 0.46573544,  0.56106597,  0.        ,  0.        ,  0.        ,
             0.        ,  0.39179133,  0.        ,  0.        ,  0.56106597,
             0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.46063063,  0.        ,
             0.        ,  0.32165752,  0.        ,  0.5709398 ,  0.46063063,
             0.        ,  0.38236504,  0.        ],
           [ 0.43078923,  0.        ,  0.        ,  0.51896668,  0.        ,
             0.        ,  0.36239348,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.64324583],
           [ 0.        ,  0.        ,  0.51530555,  0.        ,  0.51530555,
             0.        ,  0.29031416,  0.51530555,  0.        ,  0.        ,
             0.        ,  0.34510614,  0.        ],
           [ 0.35554904,  0.42832572,  0.        ,  0.        ,  0.        ,
             0.53089869,  0.        ,  0.        ,  0.        ,  0.        ,
             0.53089869,  0.35554904,  0.        ]])

