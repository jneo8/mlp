# Install tensorflow


## CPU
```bash
pip install --upgrade tensorflow
```

## GPU
```bash
pip install tensorflow-gpu
```

## test  installation

```bash
python -c 'import tensorflow; print(tensorflow.__version__)'
```

---

# Creating First Graph and Running it in a Session

```python
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='Y')

f = x * x * y + y + 2
```

This code do not actually perform any computation.
It just create a computation graph.

## tf Session

```python
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

sess.close()
```

### tf session better way

```python
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
```

## global_variables_initializer()

```python
init = tf.global_variables_initializer() # prepare an init node

with tf.Session() as sess:
    init.run()  # actually initialize all the variables
    result = f.eval()
```

## InteractiveSession

The only difference from a regular Session is that when an InteractiveSession is created it automatically sets itself as the default session, so you don't need a with block.(But you do need to cloas the session manually when you are done with it)

```python
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
```

A TensorFlow program is typically split into two parts: 

- Build a computation graph, this is called the `constructino phase`

- run it, this is the execution phase


---

# Managing Graphs

Any code you create is automatically added to the default graph

```python
x1 = tf.Variable(1)

>> x1.graph is tf.get_default_graph()
True
```

In most case this is fine, but sometimes you may want to manage multiple independent graphs.
By creating a new Graph and temporarily making the default graph inside a with block:

```python
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

>> x2.graph is graph
True

>> x2.graph is tf.get_default_graph()
False

```

---

# Lifecycle of Node Value

```python
w = tf.constant(3)
x = w + 2
y = w + 5
z = w * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(x.eval())  # 15
```

All node values are dropped between graph runs, except varible values which are maintained by the session across graph runs.

In this case evaluates x and w twice.

### more efficiently

```python
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15
```


### Tips
**In single process tensorflow, multiple sessions do not share any state, even if they reuse the same graph (each session would have its own copy of every variable). In distributed TensorFlow, variable state is stored on the servers, not in the session, so multiple session can share the same variables.**


---


# Linear Regression with Tensorflow

- ### operations
    - also called ops for short
    - Can take any number of input and produce any number of output
    - For example, the addition and multiplication ops each take two inputs and produce one output.

- ### Constants and Variables 
    - take no input
    - they are called source ops

- ### tensors
    - The inputs and outputs are multidimensional arrays.
    - Just like Numpy arrays, tensors have a type and shape, In fact, in the Python API tensors are simply represented by Numpy ndarrays. They typically contain floats, but you can also use them to carry strings(arbitary byte array).



```python
"""Linear Regression with TnesorFlow."""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing


sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confs import logconf

logger = logconf.Logger(__file__).logger


def main():
    """Main."""
    housing = fetch_california_housing()
    logger.debug(type(housing))
    m, n = housing.data.shape
    logger.debug(f'{m} {n}')
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    logger.debug(housing_data_plus_bias)
    logger.debug(housing_data_plus_bias.shape)
    x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
    xt = tf.transpose(x)
    logger.debug(xt)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt, x)), xt), y)

    with tf.Session() as sess:
        theta_value = theta.eval()
        logger.debug(theta_value)


if  __name__  == '__main__':
    main()

```


















