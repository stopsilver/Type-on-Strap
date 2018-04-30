---
layout: post
title: [핸즈온머신러닝] Chapter 9. Up and Running with TensorFlow
comments: true
tags: [ML, HandsOnMachineLearning, tensorflow]
category: ml
excerpt_separator: <!--more-->
---

이 포스트는 Hands-On Machine Learning with Scikit-Learn and TensorFlow의 내용을 요약했습니다.

# Chapter 9. Up and Running with TensorFlow
<!--more-->
이번 장에서는 Tensorflow를 이용하는 기초적인 방법들을 배웁니다.

### Tensorflow
Tensorflow는 구글에서 2015년 공개한 machine learning용 오픈소스 라이브러리입니다.

- 확장성이 뛰어나고
- C++ 기반이라 속도가 빠르며
- Windows, Linux, macOS 등의 컴퓨터 os 뿐만 아니라 mobile devices(iOS and Android)에서도 사용 가능
- TensorBoard를 이용해 쉽게 visualization 가능
- 커뮤니티가 활발함

python으로 그래프를 만든 뒤, 그 그래프를 tensorflow에 넣어주면 여러가지 이론을 이용해 자동으로 optimizing 해줍니다

- construction phase: 그래프를 만드는 단계
- execution phase: 만든 그래프를 실행시키는 단계, training step을 반복적으로 돌려 그래프(모델)의 최적 parameter 값을 구함

## Installation


```python
!pip install tensorflow
```

    Requirement already satisfied: tensorflow in c:\users\jieun\anaconda64\lib\site-packages
    Requirement already satisfied: tensorflow-tensorboard<0.5.0,>=0.4.0rc1 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow)
    Requirement already satisfied: six>=1.10.0 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow)
    Requirement already satisfied: numpy>=1.12.1 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow)
    Requirement already satisfied: protobuf>=3.3.0 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow)
    Requirement already satisfied: enum34>=1.1.6 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow)
    Requirement already satisfied: wheel>=0.26 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow)
    Requirement already satisfied: bleach==1.5.0 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow)
    Requirement already satisfied: html5lib==0.9999999 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow)
    Requirement already satisfied: werkzeug>=0.11.10 in c:\users\jieun\anaconda64\lib\site-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow)
    Requirement already satisfied: setuptools in c:\users\jieun\anaconda64\lib\site-packages\setuptools-27.2.0-py3.6.egg (from protobuf>=3.3.0->tensorflow)



```python
import tensorflow as tf
from datetime import datetime
```


```python
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
```

## How To Use

아래와 같은 방법으로 그래프를 만들 수 있다.


```python
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```

지금은 그래프만 만들어진 상태이고, 아직 계산을 하는 코드는 아니다. 계산을 하기 위해서는 Tensorflow의 Session 객체가 필요하다.

### Session
- initialize variables
- CPU나 GPU에 연산을 부여
- variable에 대한 정보를 가지고 있는다

아래 코드는 session을 만들고, variable을 initialize하고, 계산하는 과정이다


```python
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

print(result)
```

    42


위의 코드에서는 x와 y를 각각 initialize 했지만, `tf.global_variables_initializer()` 함수를 이용하면 모든 변수를 한번에 초기화할 수 있다


```python
init = tf.global_variables_initializer() # prepare an init node

with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()

print(result)
```

    42


Session 대신 InteractiveSession을 만들 수도 있다.

### InteractiveSession

- 자기 자신을 default session으로 만듬
- 그러므로 with 블록을 사용하지 않아도 됨

### Lifecycle of a Node Value

Tensorflow는 사용할 노드를 자동으로 감지한다.


```python
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15
```

    10
    15


위와 같은 코드에서 y와 z를 계산할 땐 x, w를 사용해야한다는 것을 자동으로 판단한다. 또한 eval() 함수에선 그 전 계산값을 저장하지 않는다.
즉, y를 계산할 때 x를 계산하고, z를 계산할 때 또 x를 다시 구한다는 뜻이다.

그러므로 효과적인 계산을 위해서는 y, z를 한 세션에서 실행시켜주는 것이 좋다. 아래의 코드를 참고하자.


```python
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15
```

    10
    15


## Linear Regression with TensorFlow

### Using Normal Equation 


```python
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

print(theta_value)
```

    Downloading Cal. housing from https://ndownloader.figshare.com/files/5976036 to /Users/jieun/scikit_learn_data


    [[-3.7465141e+01]
     [ 4.3573415e-01]
     [ 9.3382923e-03]
     [-1.0662201e-01]
     [ 6.4410698e-01]
     [-4.2513184e-06]
     [-3.7732250e-03]
     [-4.2664889e-01]
     [-4.4051403e-01]]


## Using Manually Computed Gradient Descent


```python
from sklearn.preprocessing import StandardScaler

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print(best_theta)
```

    Epoch 0 MSE = 9.161543
    Epoch 100 MSE = 0.7145007
    Epoch 200 MSE = 0.5667047
    Epoch 300 MSE = 0.5555716
    Epoch 400 MSE = 0.5488116
    Epoch 500 MSE = 0.54363626
    Epoch 600 MSE = 0.53962916
    Epoch 700 MSE = 0.53650916
    Epoch 800 MSE = 0.5340678
    Epoch 900 MSE = 0.53214705
    [[ 2.0685525 ]
     [ 0.8874027 ]
     [ 0.14401658]
     [-0.34770882]
     [ 0.36178368]
     [ 0.00393812]
     [-0.04269557]
     [-0.6614528 ]
     [-0.63752776]]


### Using autodiff


```python
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

gradients = tf.gradients(mse, [theta])[0]

training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
```

    Epoch 0 MSE = 9.161543
    Epoch 100 MSE = 0.7145006
    Epoch 200 MSE = 0.56670463
    Epoch 300 MSE = 0.5555716
    Epoch 400 MSE = 0.5488117
    Epoch 500 MSE = 0.5436362
    Epoch 600 MSE = 0.53962916
    Epoch 700 MSE = 0.53650916
    Epoch 800 MSE = 0.5340678
    Epoch 900 MSE = 0.53214717
    Best theta:
    [[ 2.0685525 ]
     [ 0.8874027 ]
     [ 0.14401658]
     [-0.34770882]
     [ 0.36178368]
     [ 0.00393811]
     [-0.04269556]
     [-0.6614528 ]
     [-0.6375277 ]]


### Using a GradientDescentOptimizer

가장 쉽고, 많이 사용하는 방법이다


```python
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# 만약 다른 optimizer를 사용하고싶을 때
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
```

    Epoch 0 MSE = 9.161543
    Epoch 100 MSE = 0.7145006
    Epoch 200 MSE = 0.56670463
    Epoch 300 MSE = 0.5555716
    Epoch 400 MSE = 0.5488117
    Epoch 500 MSE = 0.5436362
    Epoch 600 MSE = 0.53962916
    Epoch 700 MSE = 0.53650916
    Epoch 800 MSE = 0.5340678
    Epoch 900 MSE = 0.53214717
    Best theta:
    [[ 2.0685525 ]
     [ 0.8874027 ]
     [ 0.14401658]
     [-0.34770882]
     [ 0.36178368]
     [ 0.00393811]
     [-0.04269556]
     [-0.6614528 ]
     [-0.6375277 ]]


### Using Scikit-Learn


```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])
```

    [[-3.69419202e+01]
     [ 4.36693293e-01]
     [ 9.43577803e-03]
     [-1.07322041e-01]
     [ 6.45065694e-01]
     [-3.97638942e-06]
     [-3.78654265e-03]
     [-4.21314378e-01]
     [-4.34513755e-01]]


    /Users/jieun/anaconda/envs/tf/lib/python3.5/site-packages/scipy/linalg/basic.py:1226: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      warnings.warn(mesg, RuntimeWarning)


# Feeding data to the training algorithm

그래프에 데이터를 넣는 방법을 알아본다. 여기에서는 placeholder라는 개념을 이용하며, 많은 x와 y 데이터를 직접 넣는것이 아니고, traning interation에 따라 placeholder에 변수를 바꿔 끼워주는 의미로 생각하면 쉽다.


```python
reset_graph()

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
print(B_val_2)
```

    [[6. 7. 8.]]
    [[ 9. 10. 11.]
     [12. 13. 14.]]


이를 이용해 mini-batch GD 구현이 가능하다


```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

n_epochs = 10
learning_rate = 0.01


theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    
print(best_theta)
```

    [[ 2.0703337 ]
     [ 0.8637145 ]
     [ 0.12255151]
     [-0.31211874]
     [ 0.38510373]
     [ 0.00434168]
     [-0.01232954]
     [-0.83376896]
     [-0.8030471 ]]


# Saving and restoring a model

모델을 train시켰으면, 계산된 parameter를 저장해 나중에 다시 불러와 해당 모델을 사용할 수 있다. 모든 iteration이 끝나고 저장할 수도 있지만, train 중간에 checkpoint를 만들어 저장하는 것도 가능하다.

해당 기능에는 Savor 노드를 사용하면 된다.


```python
reset_graph()

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
error = y_pred - y                                                                    # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())                                # not shown
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)
    
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
```

    Epoch 0 MSE = 9.161543
    Epoch 100 MSE = 0.7145006
    Epoch 200 MSE = 0.56670463
    Epoch 300 MSE = 0.5555716
    Epoch 400 MSE = 0.5488117
    Epoch 500 MSE = 0.5436362
    Epoch 600 MSE = 0.53962916
    Epoch 700 MSE = 0.53650916
    Epoch 800 MSE = 0.5340678
    Epoch 900 MSE = 0.53214717



```python
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval() # not shown in the book
```

    INFO:tensorflow:Restoring parameters from /tmp/my_model_final.ckpt



```python
saver = tf.train.Saver({"weights": theta})
```

# Visualizing the Graph and Training Curves Using TensorBoard

## In Jupyter


```python
# To plot pretty figures
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tensorflow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
```


```python
show_graph(tf.get_default_graph())
```


## Using TensorBoard


```python
reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()          

file_writer.close()
```

Terminal에 다음과 같은 명령어를 입력하면 log로 저장된 파일의 그래프를 볼 수 있다.

    tensorboard --logdir tf_logs/

# Name Scopes
복잡한 NN등의 그래프를 만들다보면 그래프는 많은 노드로 인해 어지러워질 수 있다. 이를 피하기 위해 관련된 노드에 name scope를 만들 수 있다.


```python
reset_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
```


```python
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
```


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.flush()
file_writer.close()
print("Best theta:")
print(best_theta)
```

    Best theta:
    [[ 2.0703337 ]
     [ 0.8637145 ]
     [ 0.12255151]
     [-0.31211874]
     [ 0.38510373]
     [ 0.00434168]
     [-0.01232954]
     [-0.83376896]
     [-0.8030471 ]]



```python
print(error.op.name)
print(mse.op.name)
```

    loss/sub
    loss/mse



```python
show_graph(tf.get_default_graph())
```



```python

reset_graph()

a1 = tf.Variable(0, name="a")      # name == "a"
a2 = tf.Variable(0, name="a")      # name == "a_1"

with tf.name_scope("param"):       # name == "param"
    a3 = tf.Variable(0, name="a")  # name == "param/a"

with tf.name_scope("param"):       # name == "param_1"
    a4 = tf.Variable(0, name="a")  # name == "param_1/a"

for node in (a1, a2, a3, a4):
    print(node.op.name)
```

    a
    a_1
    param/a
    param_1/a


# Modularity

코드가 너무 복잡하거나, 중복으로 사용되면 모듈화를 해주자!

Before Modularity


```python
reset_graph()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0., name="relu1")
relu2 = tf.maximum(z1, 0., name="relu2")

output = tf.add(relu1, relu2, name="output")
```

After Modularity


```python
reset_graph()

def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```


```python
show_graph(tf.get_default_graph())
```



```python
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```


```python
show_graph(tf.get_default_graph())
```


# Sharing Variables
Sharing a threshold variable 

- the classic way, by defining it outside of the relu() function then passing it as a parameter

```python
def relu(X, threshold):
    with tf.name_scope("relu"):
        [...]
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")
```

    
- set the shared variable as an attribute of the relu() function upon the first call

```python
def relu(X):
    with tf.name_scope("relu"):
    if not hasattr(relu, "threshold"):
    relu.threshold = tf.Variable(0.0, name="threshold")
    [...]
    return tf.maximum(z, relu.threshold, name="max")
```
        

- TF의 get_variable() function 이용

```python
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
    initializer=tf.constant_initializer(0.0))

with tf.variable_scope("relu", reuse=True):
    threshold = tf.get_variable("threshold")

with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")
```


```python
reset_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close()
```

relu 밖에서 항상 threshold를 initialize 하는 번거로움을 피하기 위해 다음의 코드도 가능하다


```python
reset_graph()

def relu(X):
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("", default_name="") as scope:
    first_relu = relu(X)     # create the shared variable
    scope.reuse_variables()  # then reuse it
    relus = [first_relu] + [relu(X) for i in range(4)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
file_writer.close()
```


```python
reset_graph()

def relu(X):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
    b = tf.Variable(0.0, name="bias")                           # not shown
    z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []

for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")


file_writer = tf.summary.FileWriter("logs/relu9", tf.get_default_graph())
file_writer.close()
```
